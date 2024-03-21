import torch.nn as nn
import torch
from lib.config import cfg
import lib.utils.net_util as net_util


class Renderer(nn.Module):
    def __init__(self, network, upper, lower):
        super(Renderer, self).__init__()
        self.network = network
        self.upper = upper
        self.lower = lower

    def get_segment_mask(self, inp, thres_upper=0.36, thres_lower=-0.24):
        z = inp[..., 1]
        mask_head = (z >= thres_upper)
        mask_upper = (z < thres_upper) & (z > thres_lower)
        mask_lower = (z <= thres_lower)
        return mask_head, mask_upper, mask_lower

    def get_shadow_mask(self, inp, xl, xr, yl, yr):
        in_xl = inp[..., 0] > xl
        in_xr = inp[..., 0] < xr
        in_yl = inp[..., 1] > yl
        in_yr = inp[..., 1] < yr
        return in_yr & in_yl & in_xr & in_xl

    def forward(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = cfg.chunk
        all_ret = {'rgb_map': [], 'acc_map': []}
        for i in range(0, n_pixel, chunk):

            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            wpts, z_vals, viewdir, sh = net_util.get_input(ray_o_chunk, ray_d_chunk, near_chunk, far_chunk, False)

            n_batch, n_pixel, n_sample = sh
            all_wpts = wpts[None]
            pose_pts = net_util.world_points_to_pose_points(all_wpts, batch['R'], batch['Th'])
            viewdir = viewdir[None]
            pose_dirs = net_util.world_dirs_to_pose_dirs(viewdir, batch['R'])
            dist = z_vals.reshape(n_batch, n_pixel * n_sample, -1)
            with torch.no_grad():
                pind, idx, dist_weight = net_util.clamp_pts(pose_pts, batch['pvertices'])

                pose_pts = pose_pts[pind][None]

                wpts = all_wpts[pind][None]
                viewdir = viewdir[pind][None]
                pose_dirs = pose_dirs[pind][None]
                dist = dist[pind][None]

            # scale=1
            bary_coords, vert_ids, rel, fnormal = net_util.get_bary_coord(pose_pts, batch['pvertices'], batch['faces'])
            bary_coords = torch.from_numpy(bary_coords).to(pose_pts)
            vert_ids = torch.from_numpy(vert_ids).to(pose_pts.device)
            vert_ids = vert_ids.to(torch.long)
            rel = torch.from_numpy(rel).to(pose_pts)

            cnl_pts, viewdir, skinning_weight = \
                self.network.backward_deform(pose_pts, pose_dirs, None, bary_coords, vert_ids, rel, batch, 1)
            if len(bary_coords.shape) == 1:
                bary_coords = bary_coords[None]

            if cfg.type == 'reshadow':
                mask_shadow = self.get_shadow_mask(cnl_pts, -0.21, 0.4, -0.28, 0.3)
                albedo, shadow, light, alpha = self.network(viewdir,
                                                            wpts,
                                                            pose_pts,
                                                            cnl_pts,
                                                            bary_coords,
                                                            vert_ids,
                                                            rel, None,
                                                            skinning_weight,
                                                            batch['trained_param'], batch)
                if mask_shadow.sum() > 0:
                    _, tmp_shadow, _, tmp_alpha = self.upper(viewdir[mask_shadow][None],
                                                             wpts[mask_shadow][None],
                                                             pose_pts[mask_shadow][None],
                                                             cnl_pts[mask_shadow][None],
                                                             bary_coords[None][mask_shadow],
                                                             vert_ids[None][mask_shadow],
                                                             rel[None][mask_shadow], None,
                                                             skinning_weight[mask_shadow][None],
                                                             batch['upper_param'], batch)
                    ori_shadow = shadow[mask_shadow]
                    # shadow[mask_shadow]= 0.8*tmp_shadow+0.2*ori_shadow
                    shadow[mask_shadow] = tmp_shadow
                    alpha[mask_shadow] = alpha[mask_shadow] * 0.2 + 0.8 * tmp_alpha
                    # shadow[mask_shadow] = shadow[mask_shadow] ** 0.7
                    # shadow[mask_shadow]=shadow[mask_shadow]*0.7
                    # shadow[mask_shadow]*=0.9

                color = albedo * shadow * light
            else:
                mask_head, mask_upper, mask_lower = self.get_segment_mask(cnl_pts)
                if mask_head.sum() > 0:
                    head_albedo, head_shadow, head_light, head_alpha = self.network(viewdir[mask_head][None],
                                                                                    wpts[mask_head][None],
                                                                                    pose_pts[mask_head][None],
                                                                                    cnl_pts[mask_head][None],
                                                                                    bary_coords[None][mask_head],
                                                                                    vert_ids[None][mask_head],
                                                                                    rel[None][mask_head], None,
                                                                                    skinning_weight[mask_head][None],
                                                                                    batch['trained_param'], batch)

                batch['tvertices'] = batch['upper_data']['tvertices']
                batch['tbounds'] = batch['upper_data']['tbounds']
                batch['ttb'] = batch['upper_data']['ttb']

                if mask_upper.sum() > 0:
                    upper_albedo, upper_shadow, upper_light, upper_alpha = self.upper(viewdir[mask_upper][None],
                                                                                      wpts[mask_upper][None],
                                                                                      pose_pts[mask_upper][None],
                                                                                      cnl_pts[mask_upper][None],
                                                                                      bary_coords[None][mask_upper],
                                                                                      vert_ids[None][mask_upper],
                                                                                      rel[None][mask_upper], None,
                                                                                      skinning_weight[mask_upper][None],
                                                                                      batch['upper_param'], batch)
                batch['tvertices'] = batch['lower_data']['tvertices']
                batch['tbounds'] = batch['lower_data']['tbounds']
                batch['ttb'] = batch['lower_data']['ttb']
                if mask_lower.sum() > 0:
                    lower_albedo, lower_shadow, lower_light, lower_alpha = self.lower(viewdir[mask_lower][None],
                                                                                      wpts[mask_lower][None],
                                                                                      pose_pts[mask_lower][None],
                                                                                      cnl_pts[mask_lower][None],
                                                                                      bary_coords[None][mask_lower],
                                                                                      vert_ids[None][mask_lower],
                                                                                      rel[None][mask_lower], None,
                                                                                      skinning_weight[mask_lower][None],
                                                                                      batch['lower_param'], batch)

                color = torch.zeros_like(cnl_pts).to(cnl_pts)
                if mask_head.sum() > 0:
                    color[mask_head] = head_albedo * head_shadow * head_light
                if mask_upper.sum() > 0:
                    color[mask_upper] = upper_albedo * upper_shadow * upper_light
                if mask_lower.sum() > 0:
                    color[mask_lower] = lower_albedo * lower_shadow * lower_light

                alpha = torch.zeros((*cnl_pts.shape[:2], 1)).to(cnl_pts)
                if mask_head.sum() > 0:
                    alpha[mask_head] = head_alpha
                if mask_upper.sum() > 0:
                    alpha[mask_upper] = upper_alpha
                if mask_lower.sum() > 0:
                    alpha[mask_lower] = lower_alpha

            raw = torch.zeros([n_batch, n_pixel * n_sample, 1]).to(cnl_pts)
            raw[pind] = alpha
            raw = raw.reshape(n_batch, n_pixel, n_sample, -1)

            disp_map, acc_map, weights, depth_map = net_util.raw2outputs(raw, z_vals)
            tmp_rgb = torch.zeros([n_batch, n_pixel * n_sample, 3]).to(pose_pts)
            tmp_rgb[pind] = color
            tmp_rgb = tmp_rgb.reshape(n_batch, n_pixel, n_sample, -1)
            rgb_map = torch.sum(weights[..., None] * tmp_rgb, -2)
            if cfg.white_bkgd:
                rgb_map = rgb_map + torch.clip(cfg.bg_color[0] - acc_map[..., None], 0, 1)
            all_ret['rgb_map'].append(rgb_map.detach().cpu())
            all_ret['acc_map'].append(acc_map.detach().cpu())
        ret = {}

        for k in all_ret:
            ret[k] = torch.cat(all_ret[k], 1)

        return ret
