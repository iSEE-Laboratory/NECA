import numpy as np
import cv2
from lib.config import cfg


def get_rays(H, W, K, R, T,split):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)

    pixel_world = np.dot(pixel_camera - T.ravel(), R)

    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]

    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / (xyz[:, 2:]+1e-8)
    return xy


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)

    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / (ray_d[:, None]+1e-9)).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box


def sample_ray(img, mask, K, R, T, bounds, nrays, split,unsample_region_mask=None):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T,split)

    pose = np.concatenate([R, T], axis=1)

    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
    if unsample_region_mask is not None:
        bound_mask = np.logical_and(bound_mask, unsample_region_mask < 1e-6)

    if cfg.mask_bkgd:
        img[bound_mask != 1] = cfg.bg_color[0]

    mask = mask * bound_mask
    bound_mask[mask == 100] = 0

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = cfg.face_sample_ratio
        body_sample_ratio = cfg.body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_face

            # sample rays on body
            coord_body = np.argwhere(mask == 1)
            coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]
            # sample rays on face
            coord_face = np.argwhere(mask == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]
            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)

        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.argwhere(mask_at_box.reshape(H, W) == 1)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def sample_patch(rgb, mask, orig_mask,wbounds, K, R, T):
    H, W = rgb.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T,'train')

    ray_img = rgb.reshape(-1, 3)
    ray_o = ray_o.reshape(-1, 3)  # (H, W, 3) --> (N_rays, 3)
    ray_d = ray_d.reshape(-1, 3)

    near, far, mask_at_box = get_near_far(wbounds, ray_o, ray_d)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    ray_img = ray_img[mask_at_box]

    near = near[:, None].astype('float32')
    far = far[:, None].astype('float32')

    return sample_patch_rays(img=rgb, H=H, W=W,
                             subject_mask=mask,
                             orig_mask=orig_mask,
                             bbox_mask=mask_at_box.reshape(H, W),
                             ray_mask=mask_at_box,
                             rays_o=ray_o,
                             rays_d=ray_d,
                             ray_img=ray_img,
                             near=near,
                             far=far)


def sample_patch_rays(img, H, W,
                      subject_mask,
                      orig_mask,
                      bbox_mask,
                      ray_mask,
                      rays_o,
                      rays_d,
                      ray_img,
                      near,
                      far):
    select_inds, patch_info, patch_div_indices = \
        get_patch_ray_indices(
            N_patch=cfg.num_patch,
            ray_mask=ray_mask,
            subject_mask=subject_mask > 0.,
            bbox_mask=bbox_mask,
            patch_size=cfg.patch_size,
            H=H, W=W)

    rays_o = rays_o[select_inds]
    rays_d = rays_d[select_inds]
    ray_img = ray_img[select_inds]
    subject_mask = subject_mask.reshape(-1, )
    orig_mask = orig_mask.reshape(-1, )
    orig_mask=orig_mask[ray_mask][select_inds]
    ray_mask = subject_mask[ray_mask][select_inds]
    near = near[select_inds]
    far = far[select_inds]

    targets = []
    for i in range(cfg.num_patch):
        x_min, y_min = patch_info['xy_min'][i]
        x_max, y_max = patch_info['xy_max'][i]
        targets.append(img[y_min:y_max, x_min:x_max])
    target_patches = np.stack(targets, axis=0)  # (N_patches, P, P, 3)

    patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

    return rays_o, rays_d, ray_img, ray_mask, orig_mask,near, far, \
           target_patches, patch_masks, patch_div_indices


def get_patch_ray_indices(
        N_patch,
        ray_mask,
        subject_mask,
        bbox_mask,
        patch_size,
        H, W):
    assert subject_mask.dtype == bool
    assert bbox_mask.dtype == bool

    bbox_exclude_subject_mask = np.bitwise_and(
        bbox_mask,
        np.bitwise_not(subject_mask)
    )

    list_ray_indices = []
    list_mask = []
    list_xy_min = []
    list_xy_max = []

    total_rays = 0
    patch_div_indices = [total_rays]
    for _ in range(N_patch):
        # let p = cfg.patch.sample_subject_ratio
        # prob p: we sample on subject area
        # prob (1-p): we sample on non-subject area but still in bbox
        if np.random.rand(1)[0] < cfg.sample_subject_ratio:
            candidate_mask = subject_mask
        else:
            candidate_mask = bbox_exclude_subject_mask

        ray_indices, mask, xy_min, xy_max = \
            _get_patch_ray_indices(ray_mask, candidate_mask,
                                   patch_size, H, W)

        assert len(ray_indices.shape) == 1
        total_rays += len(ray_indices)

        list_ray_indices.append(ray_indices)
        list_mask.append(mask)
        list_xy_min.append(xy_min)
        list_xy_max.append(xy_max)

        patch_div_indices.append(total_rays)

    select_inds = np.concatenate(list_ray_indices, axis=0)
    patch_info = {
        'mask': np.stack(list_mask, axis=0),
        'xy_min': np.stack(list_xy_min, axis=0),
        'xy_max': np.stack(list_xy_max, axis=0)
    }
    patch_div_indices = np.array(patch_div_indices)

    return select_inds, patch_info, patch_div_indices


def _get_patch_ray_indices(
        ray_mask,
        candidate_mask,
        patch_size,
        H, W):
    assert len(ray_mask.shape) == 1
    assert ray_mask.dtype == bool
    assert candidate_mask.dtype == bool

    valid_ys, valid_xs = np.where(candidate_mask)

    # determine patch center
    select_idx = np.random.choice(valid_ys.shape[0],
                                  size=[1], replace=False)[0]
    center_x = valid_xs[select_idx]
    center_y = valid_ys[select_idx]

    # determine patch boundary
    half_patch_size = patch_size // 2
    x_min = np.clip(a=center_x - half_patch_size,
                    a_min=0,
                    a_max=W - patch_size)
    x_max = x_min + patch_size
    y_min = np.clip(a=center_y - half_patch_size,
                    a_min=0,
                    a_max=H - patch_size)
    y_max = y_min + patch_size

    sel_ray_mask = np.zeros_like(candidate_mask)
    sel_ray_mask[y_min:y_max, x_min:x_max] = True

    #####################################################
    ## Below we determine the selected ray indices
    ## and patch valid mask

    sel_ray_mask = sel_ray_mask.reshape(-1)
    inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
    select_masked_inds = np.where(inter_mask)

    masked_indices = np.cumsum(ray_mask) - 1
    select_inds = masked_indices[select_masked_inds]

    inter_mask = inter_mask.reshape(H, W)

    return select_inds, \
           inter_mask[y_min:y_max, x_min:x_max], \
           np.array([x_min, y_min]), np.array([x_max, y_max])


def get_camrot(campos, lookat=None, inv_camera=False):
    r""" Compute rotation part of extrinsic matrix from camera posistion and
         where it looks at.

    Args:
        - campos: Array (3, )
        - lookat: Array (3, )
        - inv_camera: Boolean

    Returns:
        - Array (3, 3)

    Reference: http://ksimek.github.io/2012/08/22/extrinsic/
    """

    if lookat is None:
        lookat = np.array([0., 0., 0.], dtype=np.float32)

    # define up, forward, and right vectors
    up = np.array([0., 1., 0.], dtype=np.float32)
    if inv_camera:
        up[1] *= -1.0
    forward = lookat - campos
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    camrot = np.array([right, up, forward], dtype=np.float32)
    return camrot


def setup_camera(img_size, angle, radius=10.0, focal=1250):
    x = 0
    y = -0.25
    z = radius
    campos = np.array([x, y, z, 1], dtype='float32')
    angle = 2 * np.pi * (angle / 360)
    rot = np.eye(4, 4)
    rot[0, 0] = np.cos(angle)
    rot[0, 2] = np.sin(angle)
    rot[2, 0] = -np.sin(angle)
    rot[2, 2] = np.cos(angle)

    campos = np.dot(campos, rot)[:3]
    camrot = get_camrot(campos,
                        lookat=np.array([0, y, 0.]),
                        inv_camera=True)

    E = np.eye(4, dtype='float32')
    E[:3, :3] = camrot
    E[:3, 3] = -camrot.dot(campos)

    K = np.eye(3, dtype='float32')
    K[0, 0] = focal
    K[1, 1] = focal
    K[:2, 2] = img_size / 2.
    return K, E


def get_ray_from_KRT(bounds, H, W, K, R, T):
    ray_o, ray_d = get_rays(H, W, K, R, T,'test')
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    return ray_o, ray_d, near, far, mask_at_box

def image_rays(RT, K, bounds):

    H = cfg.H * cfg.image_ratio
    W = cfg.W * cfg.image_ratio
    ray_o, ray_d = get_rays(H, W, K,
                            RT[:3, :3], RT[:3, 3],'test')

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]

    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])

    return ray_o, ray_d, near, far, center, scale, mask_at_box
