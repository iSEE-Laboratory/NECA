import igl
import numpy as np
import torch.nn.functional as F
import torch
import os
from lib.config import cfg
from pytorch3d.ops.knn import knn_points
import torch.nn as nn
import math


def func_linear2srgb(tensor):
    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor = torch.clip(tensor, 0., 1.)

    tensor_linear = tensor * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (torch.pow(tensor + 1e-7, 1 / srgb_exponent)) - (
                srgb_exponential_coeff - 1)

    is_linear = tensor <= srgb_linear_thres
    tensor_srgb = torch.where(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb


def save_model(net, optim, recorder, iter_step, name=''):
    if name == '':
        name = 'latest'
    if name != 'latest':
        if os.path.exists(os.path.join(cfg.model_dir, name + '.pth')):
            return

    model = {
        'net': (net.module.state_dict()) if cfg.distributed else (net.state_dict()),
        'optim': optim.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'iter_step': iter_step
    }
    torch.save(model, os.path.join(cfg.model_dir, name + '.pth'))


def load_model(net, optim, recorder, resume=False):
    if not resume:
        return 0
    model_dir = os.path.join(cfg.model_dir, 'latest.pth')
    if cfg.ckpt_path != '':
        model_dir = cfg.ckpt_path
    print('load model: ' + model_dir)
    model = torch.load(model_dir, 'cpu')

    net.load_state_dict(model['net'])
    optim.load_state_dict(model['optim'])
    recorder.load_state_dict(model['recorder'])
    return model['iter_step']


def load_net(net):
    model_dir = os.path.join(cfg.model_dir, 'latest.pth')

    model = torch.load(model_dir, 'cpu')

    net.load_state_dict(model['net'])

    iter_step = model['iter_step']

    return iter_step

def get_input(ray_o, ray_d, near, far, training):
    wpts, z_vals = sample_pts(ray_o, ray_d, near, far, training, cfg.num_pts)
    viewdir = ray_d
    # transform points from the world space to the pose space
    n_batch, n_pixel, n_sample = wpts.shape[:3]
    wpts = wpts.view(n_batch * n_pixel * n_sample, -1)
    viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
    viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)
    return wpts, z_vals, viewdir, (n_batch, n_pixel, n_sample)

def clamp_pts(pose_pts, verts):
    with torch.no_grad():
        pnorm, idx, dist_weight = get_knn_points(pose_pts, verts)

        pind = pnorm < cfg.norm_th

        pind[torch.arange(len(pnorm)), pnorm.argmin(dim=0)] = True
    return pind, idx, dist_weight


def get_voxel_coord(pts, bound):
    min_xyz = bound[..., 0, :]
    max_xyz = bound[..., 1, :]
    dhw = pts[..., [2, 1, 0]]
    min_dhw = min_xyz[..., [2, 1, 0]]
    max_dhw = max_xyz[..., [2, 1, 0]]
    voxel_size = torch.tensor(cfg.voxel_size).to(pts.device)
    coord = torch.round((dhw - min_dhw) / voxel_size).type(torch.int32)

    sh = coord.shape
    idx = [torch.full([sh[1]], i) for i in range(sh[0])]
    idx = torch.cat(idx).to(coord)
    coord = coord.view(-1, sh[-1])
    coord = torch.cat([idx[:, None], coord], dim=1)

    out_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).type(torch.int32)
    x = 32
    out_sh = (out_sh | (x - 1)) + 1
    return coord, out_sh.view(-1).tolist()


def get_grid_coord(pts, bound, sh):
    # convert xyz to the voxel coordinate dhw
    dhw = pts[..., [2, 1, 0]]

    min_dhw = bound[:, 0, [2, 1, 0]]
    dhw = dhw - min_dhw[:, None]
    dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
    # convert the voxel coordinate to [-1, 1]
    out_sh = torch.tensor(sh).to(dhw)

    dhw = dhw / out_sh * 2 - 1
    # convert dhw to whd, since the occupancy is indexed by dhw
    grid_coords = dhw[..., [2, 1, 0]]
    return grid_coords[:, None, None]


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.contiguous().view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def get_gradient(y, x):
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(outputs=y,
                                    inputs=x,
                                    grad_outputs=d_output,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    return gradients


def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(
        y.device)  # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[..., i].view(-1, 1)
        jac[:, :, i, :] = \
            torch.autograd.grad(y_flat, x, torch.ones_like(y_flat), retain_graph=True, create_graph=False)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status


def sample_pts(ray_o, ray_d, near, far, training, num_pts):
    # calculate the steps for each ray
    t_vals = torch.linspace(0., 1., steps=num_pts).to(near)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if training:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(upper)
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

    return pts, z_vals


def sdf_to_alpha(sdf, beta):
    x = -sdf

    # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
    ind0 = x <= 0
    val0 = 1 / beta * (0.5 * torch.exp(x[ind0] / beta))

    # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
    ind1 = x > 0
    val1 = 1 / beta * (1 - 0.5 * torch.exp(-x[ind1] / beta))

    val = torch.zeros_like(sdf)
    val[ind0] = val0
    val[ind1] = val1

    return val


def world_points_to_pose_points(wpts, Rh, Th):
    """
    wpts: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(wpts - Th, Rh)
    return pts


def world_dirs_to_pose_dirs(wdirs, Rh):
    """
    wdirs: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    """
    pts = torch.matmul(wdirs, Rh)
    return pts


def pose_points_to_world_points(ppts, Rh, Th):
    """
    ppts: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(ppts, Rh.transpose(1, 2)) + Th
    return pts


def pose_dirs_to_world_dirs(dirs, Rh):
    """
    wdirs: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    """
    pts = torch.matmul(dirs, Rh.transpose(1, 2))
    return pts


def pose_points_to_tpose_points(ppts, bw, A):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_points, 3
    bw: n_batch, n_points, 24
    A: n_batch, 24, 4, 4
    """
    sh = ppts.shape
    A = torch.bmm(bw, A.view(sh[0], cfg.joint_num, -1))
    A = A.view(sh[0], -1, 4, 4)
    pts = ppts - A[..., :3, 3]
    R_inv = torch.inverse(A[..., :3, :3])
    pts = torch.sum(R_inv * pts[:, :, None], dim=3)
    return pts


def pose_dirs_to_tpose_dirs(ddirs, bw, A):
    """transform directions from the pose space to the T pose
    ddirs: n_batch, n_points, 3
    bw: n_batch, n_points, 24
    A: n_batch, 24, 4, 4
    """
    sh = ddirs.shape
    A = torch.bmm(bw, A.view(sh[0], cfg.joint_num, -1))
    A = A.view(sh[0], -1, 4, 4)
    R_inv = torch.inverse(A[..., :3, :3])
    pts = torch.sum(R_inv * ddirs[:, :, None], dim=3)
    return pts


def tpose_points_to_pose_points(pts, bw, A):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, n_points, 24
    A: n_batch, 24, 4, 4
    """
    sh = pts.shape
    A = torch.bmm(bw, A.view(sh[0], cfg.joint_num, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    pts = pts + A[..., :3, 3]
    return pts


def tpose_dirs_to_pose_dirs(ddirs, bw, A):
    """transform directions from the T pose to the pose space
    ddirs: n_batch, n_points, 3
    bw: n_batch, n_points, 24
    A: n_batch, 24, 4, 4
    """
    sh = ddirs.shape
    A = torch.bmm(bw, A.view(sh[0], cfg.joint_num, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * ddirs[:, :, None], dim=3)
    return pts


def get_bary_coord(pts, verts, faces):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    pts = pts[0]
    verts = verts[0]
    faces = faces[0]

    closest_dists, closest_faces, closest_points = igl.point_mesh_squared_distance(pts, verts,
                                                                                   faces)

    if pts.shape[0] == 1:
        closest_points = closest_points[None]
        closest_faces = closest_faces[None]
        closest_dists = closest_dists[None]

    v0 = verts[faces[closest_faces, 0], :]
    v1 = verts[faces[closest_faces, 1], :]
    v2 = verts[faces[closest_faces, 2], :]
    fnormal = np.cross(v1 - v0, v2 - v0)

    fnormal = fnormal / (np.linalg.norm(fnormal, axis=-1, keepdims=True) + 1e-9)

    bary_coords = igl.barycentric_coordinates_tri(closest_points, v0, v1, v2)

    vert_ids = faces[closest_faces, ...]

    dir = pts - closest_points
    v2v0 = v0 - closest_points
    return bary_coords, vert_ids, np.concatenate([dir, v2v0], -1), fnormal


def get_bary_weights(pts, verts, faces, weights, bary_coords, vert_ids):
    if not isinstance(bary_coords, torch.Tensor):
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
    weights = weights[0]
    # bary_coords, vert_ids, _, _ = get_bary_coord(pts, verts, faces)
    bw = (weights[vert_ids] * bary_coords[..., np.newaxis]).sum(axis=1)
    bw[bw < 0] = 0

    return bw


def warp_by_projection(pts_world, batch, ray_d_W=None, floor=-4, ceil=5):
    # get world meshes
    xyz = batch["pvertices"]
    xyz = xyz.to(pts_world)
    meshes = xyz[0, batch['faces'], :]
    # calculate closest mesh from pts_world
    closest_meshes, idx = get_closest_mesh(pts_world, meshes)
    # project pts_world to closest mesh
    uv, signed_distance = project_point2mesh(
        pts_world.reshape(-1, 3), meshes=closest_meshes.reshape(-1, 3, 3)
    )

    # for the clamped points, their density should be zero
    transparent_mask = get_transparent_mask(uv, signed_distance)
    # get the mapped canonical pts
    meshes_can = batch['tpose_mesh'][0, idx.flatten(), ...]
    pts_smpl_can = barycentric_map2can(uv, signed_distance, meshes_can)
    if ray_d_W != None:
        pts_ray_d_W = pts_world + ray_d_W
        uv, signed_distance = project_point2mesh(
            pts_ray_d_W.reshape(-1, 3), meshes=closest_meshes.reshape(-1, 3, 3)
        )
        pts_ray_d_can = barycentric_map2can(uv, signed_distance, meshes_can)
        ray_d_can = torch.nn.functional.normalize(
            pts_ray_d_can - pts_smpl_can, dim=-1
        )
        return pts_smpl_can, transparent_mask, ray_d_can

    else:
        return pts_smpl_can, transparent_mask


def get_knn_points(src, ref, K=5):
    ret = knn_points(src, ref, K=K)
    dists, idx = ret.dists.sqrt(), ret.idx
    disp = 1 / (dists + 1e-8)
    weight = disp / disp.sum(dim=-1, keepdim=True)
    dists = torch.einsum('ijk,ijk->ij', dists, weight)
    return dists, idx, weight


def get_closest_mesh(vsrc, meshes):
    """get closest mesh by barycentric points of each mesh

    Args:
        vsrc ([type]): [description]
        meshes ([type]): [description]

    Returns:
        [type]: [description]
    """
    mesh_centroid = meshes.mean(dim=-2)
    dist, idx, Vnn = knn_points(vsrc, mesh_centroid, K=1, return_nn=True)
    closest_meshes = torch.gather(
        meshes, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, 3, 3)
    )
    return closest_meshes, idx


def get_barycentric_coordinates(pts_proj, meshes):
    v0 = meshes[..., 2, :] - meshes[..., 0, :]
    v1 = meshes[..., 1, :] - meshes[..., 0, :]
    v2 = pts_proj - meshes[..., 0, :]

    dot00 = (v0 * v0).sum(-1)
    dot01 = (v0 * v1).sum(-1)
    dot02 = (v0 * v2).sum(-1)
    dot11 = (v1 * v1).sum(-1)
    dot12 = (v1 * v2).sum(-1)

    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    uv = torch.stack([u, v], dim=-1)
    return uv


def project_point2mesh(pts, meshes):
    """project points to corresponding meshes,points number must be the same as mesh number

    Args:
        pts (tensor): [n,3]
        meshes (tensor): [n,3,3]
    """
    assert (
            pts.shape[0] == meshes.shape[0]
    ), "points number must be the same as mesh number"
    v10 = meshes[:, 1] - meshes[:, 0]
    v20 = meshes[:, 2] - meshes[:, 0]
    normal_f = torch.cross(v10, v20)
    normal_f = normal_f / torch.norm(normal_f, dim=-1, keepdim=True)
    tmp = pts - meshes[:, 0]
    signed_distance = torch.einsum("ij,ij->i", tmp, normal_f)
    pts_proj = pts - normal_f * signed_distance.unsqueeze(-1)
    uv = get_barycentric_coordinates(pts_proj=pts_proj, meshes=meshes)

    return uv, signed_distance


def get_transparent_mask(uv, signed_distance, floor=-4, ceil=5, max_dist=0.1):
    clamped_uv_mask = torch.logical_or(uv > ceil, uv < floor)
    transparent_mask = torch.logical_or(clamped_uv_mask[:, 0], clamped_uv_mask[:, 1])
    transparent_mask = torch.logical_or(
        transparent_mask, signed_distance.abs() > max_dist
    )
    return transparent_mask


def barycentric_map2can(uv, signed_distance, meshes_can):
    """map points to canonical space by "uv and distance" coordinate

    Args:
        uv ([type]): [description]
        signed_distance ([type]): [description]
        meshes_can ([type]): [description]

    Returns:
        [type]: [description]
    """
    v2 = meshes_can[..., 2, :] - meshes_can[..., 0, :]
    v1 = meshes_can[..., 1, :] - meshes_can[..., 0, :]
    normal_f = torch.cross(v1, v2)
    normal_f = normal_f / torch.norm(normal_f, dim=-1, keepdim=True)
    offset_vec = signed_distance.unsqueeze(-1) * normal_f
    pts_proj_can = meshes_can[..., 0, :] + uv[:, 0, None] * v2 + uv[:, 1, None] * v1
    pts_smpl_can = pts_proj_can + offset_vec
    return pts_smpl_can


def acc_raw(alpha, z_vals):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
        raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, dists[..., -1:]], dim=2)
    alpha = raw2alpha(alpha, dists)

    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((*alpha.shape[:2], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[..., :-1]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return disp_map, acc_map, weights, depth_map


def raw2outputs(raw, z_vals):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_batch, num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_batch, num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
        raw) * dists)

    # rgb = raw[..., :-1]  # [N_rays, N_samples, 3]
    alpha = raw[..., -1]

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, dists[..., -1:]], dim=2)
    alpha = raw2alpha(alpha, dists)

    # rgb = torch.sigmoid(rgb)

    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((*alpha.shape[:2], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[..., :-1]
    # rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return disp_map, acc_map, weights, depth_map

def xaviermultiplier(m, gain):
    """
        Args:
            m (torch.nn.Module)
            gain (float)

        Returns:
            std (float): adjusted standard deviation
    """
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] \
                // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] \
                // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    """ Set module weight values with a uniform distribution.

        Args:
            m (torch.nn.Module)
            gain (float)
    """
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * math.sqrt(3.0)), std * math.sqrt(3.0))


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    """ Initialized module weights.

        Args:
            m (torch.nn.Module)
            gain (float)
            weightinitfunc (function)
    """
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :,
                                                0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :,
                                                0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :,
                                                0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :,
                                                0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :,
                                                0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :,
                                                0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :,
                                                0::2, 0::2, 0::2]


def initseq(s):
    """ Initialized weights of all modules in a module sequence.

        Args:
            s (torch.nn.Sequential)
    """
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])


class RodriguesModule(nn.Module):
    def forward(self, rvec):
        r''' Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)

            Returns
                rmtx: Tensor (B, 3, 3)
        '''
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh),
            dim=1).view(-1, 3, 3)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def knn_gather(x, idx):
    """
    :param x: (B, N, C)
    :param idx: (B, N, K)
    :return: (B, N, K, C)
    """
    C = x.shape[-1]
    B, N, K = idx.shape
    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, C)
    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)

    return x_out
