import os

from lib.config import cfg
import numpy as np
import cv2


class Visualizer:
    def __init__(self):
        pass

    def visualize(self, out, batch):
        rgb_pred=out['rgb_map']

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)
        default_value = cfg.bg_color[0]
        # convert the pixels into an image
        img_pred = np.zeros((H, W, 3))+default_value
        img_pred[mask_at_box] = rgb_pred
        result_dir = cfg.eval_dir
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch['frame_index'].item()
        acc_pred = np.zeros((H, W, 1))
        acc_pred[mask_at_box] = np.clip(out['acc_map'][..., None], 0., 1.)

        img_cat = np.concatenate([img_pred], axis=1)

        if 'light_map' in out:
            img_light = np.zeros((H, W, 3)) + default_value
            img_light[mask_at_box] = out['light_map'][0].detach().cpu().numpy()
            img_cat = np.concatenate([img_light, img_cat], axis=1)

        if 'shadow_map' in out:
            img_vis = np.zeros((H, W, 3)) + default_value
            img_vis[mask_at_box] = out['shadow_map'][0].detach().cpu().numpy()
            img_cat = np.concatenate([img_vis, img_cat], axis=1)

        if 'albedo_map' in out:
            img_albedo = np.zeros((H, W, 3)) + default_value
            img_albedo[mask_at_box] = out['albedo_map'][0].detach().cpu().numpy()
            img_cat = np.concatenate([img_albedo, img_cat], axis=1)
        if 'normal_map' in out:
            img_normal = np.zeros((H, W, 3)) + default_value
            img_normal[mask_at_box] = out['normal_map'][0].detach().cpu().numpy()
            img_cat = np.concatenate([img_normal, img_cat], axis=1)

        cv2.imwrite(
            '{}/{:04d}.png'.format(result_dir, frame_index),
            (img_cat[..., [2, 1, 0]] * 255))

