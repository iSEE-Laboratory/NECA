import numpy as np
from lib.config import cfg
from skimage.metrics import structural_similarity as compare_ssim
import os
import cv2
from termcolor import colored
import json
import torch


class Evaluator:
    def __init__(self, lpips):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.lp_fn = lpips
        self.path = cfg.eval_dir

    def mse_and_psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return mse, psnr

    def ssim_and_lpips_metric(self, rgb_pred, rgb_gt, batch, output):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)
        default_value = cfg.bg_color[0]
        # convert the pixels into an image
        img_pred = np.zeros((H, W, 3)) + default_value
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3)) + default_value
        img_gt[mask_at_box] = rgb_gt
        acc_pred = np.zeros((H, W, 1))
        acc_pred[mask_at_box] = np.clip(output['acc_map'][..., None], 0., 1.)
        orig_img_pred = img_pred.copy()
        orig_img_gt = img_gt.copy()

        result_dir = os.path.join(self.path, 'images')
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_index'].item()
        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = orig_img_pred[y:y + h, x:x + w]
        img_gt = orig_img_gt[y:y + h, x:x + w]

        img_cat = np.concatenate([orig_img_pred, orig_img_gt], axis=1)

        if 'light_map' in output:
            img_light = np.zeros((H, W, 3)) + default_value
            img_light[mask_at_box] = output['light_map'][0].detach().cpu().numpy()
            img_cat = np.concatenate([img_light, img_cat], axis=1)

        if 'shadow_map' in output:
            img_vis = np.zeros((H, W, 3)) + default_value
            img_vis[mask_at_box] = output['shadow_map'][0].detach().cpu().numpy()
            img_cat = np.concatenate([img_vis, img_cat], axis=1)

        if 'albedo_map' in output:
            img_albedo = np.zeros((H, W, 3)) + default_value
            img_albedo[mask_at_box] = output['albedo_map'][0].detach().cpu().numpy()
            img_cat = np.concatenate([img_albedo, img_cat], axis=1)
        if 'normal_map' in output:
            img_normal = np.zeros((H, W, 3)) + default_value
            img_normal[mask_at_box] = output['normal_map'][0].detach().cpu().numpy()
            img_cat = np.concatenate([img_normal, img_cat], axis=1)


        cv2.imwrite(
            '{}/{:04d}_view{:04d}.png'.format(result_dir, frame_index,view_index),
            (img_cat[..., [2, 1, 0]] * 255))

        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index),
            (orig_img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                      view_index),
            (orig_img_gt[..., [2, 1, 0]] * 255))

        # compute the ssim

        ssim = compare_ssim(img_pred, img_gt, channel_axis=2)
        img_pred = torch.from_numpy(img_pred).permute(2, 0, 1).to(torch.float32).cuda()
        img_gt = torch.from_numpy(img_gt).permute(2, 0, 1).to(torch.float32).cuda()
        lpips = self.lp_fn(img_pred, img_gt).squeeze().detach().cpu().numpy()

        return ssim, lpips

    def evaluate(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        if rgb_gt.sum() == 0:
            return

        mse, psnr = self.mse_and_psnr_metric(rgb_pred, rgb_gt)
        self.mse.append(float(mse))
        self.psnr.append(float(psnr))

        ssim, lpips = self.ssim_and_lpips_metric(rgb_pred, rgb_gt, batch, output)
        self.ssim.append(float(ssim))
        self.lpips.append(float(lpips.tolist()))

    def summarize(self):
        print(
            colored('the results are saved at {}'.format(cfg.eval_dir),
                    'yellow'))

        mse_mean = np.mean(self.mse)
        psnr_mean = np.mean(self.psnr)
        ssim_mean = np.mean(self.ssim)
        lpips_mean = np.mean(self.lpips)
        print('mse: {}'.format(mse_mean))
        print('psnr: {}'.format(psnr_mean))
        print('ssim: {}'.format(ssim_mean))
        print('lpips: {}'.format(lpips_mean))

        result_path = os.path.join(cfg.eval_dir, 'metrics.json')
        metrics = {
            'mean': {
                'mse': mse_mean,
                'psnr': psnr_mean,
                'ssim': ssim_mean,
                'lpips': lpips_mean
            }}
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f)
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []

