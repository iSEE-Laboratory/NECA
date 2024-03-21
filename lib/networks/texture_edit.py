import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils import net_util
from lib.networks.modules import *
import pytorch3d.transforms as tfm

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.albedo_latent = nn.Embedding(cfg.vert_num, 16)

        self.embedder, embedder_dim = get_embedder(6, 3)

        self.sdf_network = SDFNetwork(**cfg.network.sdf)
        self.beta_network = BetaNetwork(**cfg.network.beta)
        self.albedo = ColorNetwork(**cfg.network.albedo)
        self.shadow = ColorNetwork(**cfg.network.shadow)

        self.light = EnvMap(hdr_path=cfg.hdr_path)

        self.pose_num = cfg.key_pose_num

        self.mat_mode = [[0, 1], [0, 2], [1, 2]]
        self.vec_mode = [0, 1, 2]

        self.grid_size = [512, 512, 128]
        self.n_comp = [48, 48, 48]
        self.pose_dim = 32

        self.init_svd_volume()

    def init_svd_volume(self):
        self.coord_line, self.feat_line = self.init_one_svd(self.n_comp, self.grid_size, 0.1)

    def init_one_svd(self, n_component, gridSize, scale):
        line_coef = []
        feat_coef = []
        for i in range(len(self.vec_mode)):
            vec_id = self.vec_mode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((self.pose_num, n_component[i], gridSize[vec_id], 1))))
            feat_coef.append(
                torch.nn.Parameter(scale * torch.randn((self.pose_num, n_component[i], self.pose_dim, 1)))
            )
        return torch.nn.ParameterList(line_coef), torch.nn.ParameterList(feat_coef)

    def get_svd_feat(self, xyz_sampled, idx, k):
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vec_mode[0]],
             xyz_sampled[..., self.vec_mode[1]],
             xyz_sampled[..., self.vec_mode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)

        coord = []
        feat = []

        for idx_line in range(len(self.coord_line)):
            # The current F.grif_sample in pytorch doesn't support backpropagation

            line_coef_point = net_util.grid_sample(self.coord_line[idx_line][idx[0]],
                                                   coordinate_line[[idx_line]].repeat(k, 1, 1, 1)).view(k, -1,
                                                                                                        *xyz_sampled.shape[
                                                                                                         :2])  # [16, h*w*n]
            feat_coef = self.feat_line[idx_line][idx[0]].repeat(1, 1, 1, xyz_sampled.shape[1])
            coord.append(line_coef_point)
            feat.append(feat_coef)
        pose_features = []

        for idx_plane in range(len(self.mat_mode)):
            id0, id1 = self.mat_mode[idx_plane]

            plane = torch.sum(coord[id0] * coord[id1] * feat[idx_plane], dim=1)
            pose_features.append(plane)

        pose_feature = torch.cat(pose_features, dim=1)
        return pose_feature.permute(0, 2, 1)

    def query_sdf(self, x, pose_feat, code):

        if cfg.network.sdf.in_pose_feat:
            if len(pose_feat.shape) != len(x.shape):
                pose_feat = pose_feat[:, None].repeat(1, x.shape[1], 1)
        else:
            pose_feat = None
        if code is not None:
            code = torch.cat([pose_feat, code], dim=-1)
        else:
            code = pose_feat
        sdf = self.sdf_network(x, code)
        return sdf[..., :1], sdf[..., 1:]

    def get_smpl_skinning_weight(self, pts, target, bary_coords, vert_ids, batch):

        skinning_weight = net_util.get_bary_weights(pts, target, batch['faces'],
                                                    batch['weights'], bary_coords, vert_ids)[None]
        if not isinstance(skinning_weight, torch.Tensor):
            skinning_weight = torch.from_numpy(skinning_weight).to(batch['weights'])

        return skinning_weight

    def backward_deform(self, pts, pose_dirs, skinning_weight, bary_coords, vert_ids, rel, batch, scale):

        # transform points from i to i_0
        if skinning_weight is None:
            skinning_weight = self.get_smpl_skinning_weight(pts, batch['pvertices'], bary_coords, vert_ids, batch)

        init_tpose = net_util.pose_points_to_tpose_points(pts, skinning_weight,
                                                          batch['A'])
        init_bigpose = net_util.tpose_points_to_pose_points(init_tpose, skinning_weight,
                                                            batch['big_A'])

        tpose_dirs = net_util.pose_dirs_to_tpose_dirs(pose_dirs, skinning_weight,
                                                      batch['A'])
        tpose_dirs = net_util.tpose_dirs_to_pose_dirs(tpose_dirs, skinning_weight,
                                                      batch['big_A'])

        return init_bigpose, tpose_dirs, skinning_weight

    def shaped_deform(self, pts, batch, scale):
        bary_coords, vert_ids, rel, fnormal = net_util.get_bary_coord(pts, batch['shaped_tvertices'],
                                                                      batch['faces'])
        bary_coords = torch.from_numpy(bary_coords).to(pts)
        vert_ids = torch.from_numpy(vert_ids).to(pts.device)
        vert_ids = vert_ids.to(torch.long)
        rel = torch.from_numpy(rel).to(pts)

        shaped_ttb = (batch['shaped_ttb'][0][vert_ids] * bary_coords[..., None, None]).sum(axis=1)
        shaped_t_tbn = self.get_tbn(shaped_ttb)
        dir = rel[..., :3]
        dir = dir * scale
        local_coords = torch.sum(shaped_t_tbn * dir[:, None, :], dim=-1).to(pts.dtype)

        ttb = (batch['ttb'][0][vert_ids] * bary_coords[..., None, None]).sum(axis=1)
        t_tbn = self.get_tbn(ttb)

        t_rel = torch.sum(t_tbn * local_coords[:, :, None], dim=-2).to(pts.dtype)
        cloest_pts = (batch['tvertices'][0][vert_ids] * bary_coords[..., None]).sum(axis=1)
        cnl_pts = t_rel + cloest_pts
        return cnl_pts[None]

    def cnl_forward(self, pts, viewdir, local_pose_feat, latent, return_shadow=True):

        viewdir = F.normalize(viewdir)

        pts.requires_grad_(True)
        with torch.enable_grad():
            sdf, sdfeat = self.query_sdf(pts, local_pose_feat, latent)
            # calculate normal
            normal = net_util.get_gradient(sdf, pts)

        pts = pts.detach()
        # calculate alpha
        beta = self.beta_network().clamp(1e-9, 1e6)
        alpha = net_util.sdf_to_alpha(sdf, beta)

        albedo = self.albedo(pts if cfg.network.albedo.in_pts else None,
                             normal if cfg.network.albedo.in_normal else None,
                             viewdir if cfg.network.albedo.in_viewdir else None, sdfeat)

        if return_shadow:

            shadow = self.shadow(pts if cfg.network.shadow.in_pts else None,
                                 normal if cfg.network.shadow.in_normal else None,
                                 viewdir if cfg.network.shadow.in_viewdir else None, sdfeat)

            raw = torch.cat((albedo, shadow, alpha), dim=-1)
        else:
            raw = torch.cat([albedo, alpha], dim=-1)

        ret = {'raw': raw, 'gradients': normal, 'sdf': sdf}

        return ret

    def forward_deform(self, x, skinning_weight, batch, dir=False):
        if dir:
            x = net_util.pose_dirs_to_tpose_dirs(x, skinning_weight, batch['big_A'])
            x = net_util.tpose_dirs_to_pose_dirs(x, skinning_weight, batch['A'])
            x = net_util.pose_dirs_to_world_dirs(x, batch['R'])
        else:
            x = net_util.pose_points_to_tpose_points(x, skinning_weight, batch['big_A'])
            x = net_util.tpose_points_to_pose_points(x, skinning_weight, batch['A'])
            x = net_util.pose_points_to_world_points(x, batch['R'], batch['Th'])
        return x

    def get_tbn(self, tb):
        tan = F.normalize(tb[:, 0], dim=-1)
        # bitan=F.normalize(tb[:,1],dim=-1)
        normal = F.normalize(tb[:, 2], dim=-1)
        tan = (tan - torch.sum(tan * normal, dim=-1, keepdim=True) * normal)
        tan = F.normalize(tan, dim=-1)
        bitan = torch.cross(normal, tan)
        tbn = torch.stack([tan, bitan, normal], dim=-2)
        return tbn

    def get_latent(self, pose_pts, bary_coords, vert_ids, rel, fnormal, batch, scale):
        albedo_feat = self.albedo_latent(torch.arange(0, 6890, device=pose_pts.device))

        if len(bary_coords.shape) == 1:
            bary_coords = bary_coords[None]

        albedo_latent = (albedo_feat[vert_ids] * bary_coords[..., None]).sum(axis=1)

        ptb = (batch['ptb'][0][vert_ids] * bary_coords[..., None, None]).sum(axis=1)
        p_tbn = self.get_tbn(ptb)

        dir = rel[..., :3]

        dir = dir * scale
        local_coords = torch.sum(p_tbn * dir[:, None, :], dim=-1).to(pose_pts.dtype)

        emdb_local_coords = self.embedder(local_coords)

        albedo_latent = torch.cat([albedo_latent, emdb_local_coords], -1)

        return albedo_latent[None], local_coords

    def get_pose_feat(self, pts, trained_poses, batch):

        k = cfg.knn
        pose = batch['poses'][..., 3:]
        key_pose = trained_poses[..., 3:]

        key_quat = tfm.axis_angle_to_quaternion(key_pose.reshape((*key_pose.shape[:2], 23, 3)))
        query_quat = tfm.axis_angle_to_quaternion(pose.reshape(1, 23, 3))
        pose_dist = torch.abs((query_quat[:, None, :] * key_quat).sum(-1)).sum(-1)

        topk_weight, topk_id = torch.topk(pose_dist, k, dim=-1)  # (B, J, K)

        topk_weight = F.normalize(topk_weight, dim=-1, p=1, eps=1e-16)
        pts = self.normalize_pts(pts, batch['tbounds'][0])

        pose_feats = self.get_svd_feat(pts, topk_id, k)
        pose_feat = torch.sum(topk_weight[0, :, None, None] * pose_feats, dim=0, keepdim=True)
        if cfg.concat_pose:
            pose=key_pose[:,topk_id[:,0]].repeat(1, pts.shape[1], 1)
            pose_feat = torch.cat([pose_feat, pose], dim=-1)

        return pose_feat

    def normalize_pts(self, pts, bounds):
        pts = pts.detach()
        xyz_min = bounds[0]
        xyz_max = bounds[1]
        normalized_pts = (pts - xyz_min) / (xyz_max - xyz_min) * 2 - 1
        return normalized_pts

    def forward(self, viewdir, wpts, pose_pts, cnl_pts, bary_coords, vert_ids, rel, fnormal, skinning_weight,
                trained_poses, batch):

        tbound = batch['tbounds']
        shaped_tbounds = batch['shaped_tbounds']

        scale = (tbound[:, 1] - tbound[:, 0]) / (shaped_tbounds[:, 1] - shaped_tbounds[:, 0])

        albedo_latent, local_coords = self.get_latent(pose_pts, bary_coords, vert_ids, rel, fnormal, batch,
                                                      scale)

        cnl_pts = self.shaped_deform(cnl_pts, batch, scale)
        local_pose_feat = self.get_pose_feat(cnl_pts, trained_poses, batch)

        ret = self.cnl_forward(cnl_pts, viewdir, local_pose_feat, albedo_latent)

        albedo = ret['raw'][..., :3]
        shadow = ret['raw'][..., 3:4]
        alpha = ret['raw'][..., -1:]
        albedo = torch.sigmoid(albedo)
        shadow = torch.sigmoid(shadow)

        grad_normal = ret['gradients']
        grad_normal = self.forward_deform(grad_normal, skinning_weight, batch, True)
        grad_normal = F.normalize(grad_normal, dim=-1)

        light = self.light(None, wpts, grad_normal, None) / torch.pi

        return albedo, shadow, light, alpha
