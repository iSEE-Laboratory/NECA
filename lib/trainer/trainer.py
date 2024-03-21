import torch
import time
from termcolor import colored
from lib.config import cfg
import lib.utils.trainer_util as trainer_util
import lib.utils.loss_util as loss_util
import tqdm

from lib.utils import net_util

class Trainer:
    def __init__(self, device, lpips):

        self.device = device
        self.lpips = lpips
        self.start_time = time.time()
        self.iter = 0
        self.evaluator=None
        self.val_loader=None

    def compute_loss(self, out, batch, iter_step):
        loss_cfg = cfg.loss
        loss = torch.tensor(0).to(out['rgb_map'])
        loss_dict = {}
        device = out['rgb_map'].device

        img_loss = loss_util.mse(out['rgb_map'], batch['rgb'].to(device))
        loss_dict.update({'img_loss': img_loss})
        loss += loss_cfg.img * img_loss

        if ('tv_loss' in out) and loss_cfg.tv!=0:
            tv_loss = out['tv_loss'].mean()
            loss_dict.update({'tv_loss': tv_loss})
            loss += loss_cfg.tv * tv_loss

        if ('grad_normal' in out) and loss_cfg.normal!=0:
            tnormal = out['tnormal']
            grad_normal = out['grad_normal']
            grad_normal_loss = torch.maximum(-torch.sum(tnormal * grad_normal, dim=-1)+0.5, torch.tensor(0).to(grad_normal))
            grad_normal_loss = grad_normal_loss.mean()
            loss_dict.update({'grad_normal_loss': grad_normal_loss})
            loss += loss_cfg.normal * grad_normal_loss

        if ('gradients' in out) and (loss_cfg.eikonal != 0):
            gradients = out['gradients']
            grad_loss = loss_util.regularize(gradients, 1, 2)
            loss_dict.update({'grad_loss': grad_loss})
            loss += loss_cfg.eikonal * grad_loss

        if ('observed_gradients' in out) and (loss_cfg.eikonal != 0):
            ogradients = out['observed_gradients']
            ograd_loss = loss_util.regularize(ogradients, 1, 2)
            loss_dict.update({'ograd_loss': ograd_loss})
            loss += loss_cfg.eikonal * ograd_loss

        if ('sdf' in out) and (loss_cfg.mask != 0):

            mask_loss = loss_util.sdf_mask_loss(out['sdf'], batch['occ'].to(device), iter_step)
            loss_dict.update({'mask_loss': mask_loss})
            loss += loss_cfg.mask * mask_loss

        # if ('acc_map' in out) and (loss_cfg.mask != 0):
        #     # ipdb.set_trace()
        #     mask_loss = loss_util.alpha_mask_loss(out['acc_map'], batch['occ'].to(device), iter_step)
        #     loss_dict.update({'mask_loss': mask_loss})
        #     loss += loss_cfg.mask * mask_loss

        if cfg.ray_mode == 'patch' and loss_cfg.lpips != 0:
            patch_label = batch['target_patches'][0]
            patch_masks = batch['patch_masks'][0]
            div_indices = batch['patch_div_indices'][0]

            lpips_loss = loss_util.lpips_crit(out['rgb_map'][0], patch_masks, patch_label, div_indices, self.lpips)
            loss += loss_cfg.lpips * lpips_loss
            loss_dict.update({'lpips_loss': lpips_loss})

        loss_dict.update({'loss': loss})
        return loss, loss_dict

    def train(self, epoch, loader, renderer, optimizer, scheduler, recorder):
        renderer.train()
        end = time.time()
        epoch_begin = time.time()
        print(colored('Local Rank: {}, Epoch: {}'.format(cfg.local_rank, epoch), 'red'))
        for i, batch in enumerate(loader):
            data_time = time.time() - end

            batch = trainer_util.to_cuda(batch, self.device)
            trainer_util.add_iter_step(batch, self.iter)
            out = renderer(batch)
            loss, loss_dict = self.compute_loss(out, batch, self.iter)
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            lr = optimizer.param_groups[0]['lr']

            recorder.step = self.iter
            loss_dict = trainer_util.reduce_loss_stats(loss_dict)
            recorder.update_loss_stats(loss_dict)

            batch_time = time.time() - end
            end = time.time()
            epoch_time = time.time() - epoch_begin

            recorder.batch_time = batch_time
            recorder.data_time = data_time
            recorder.epoch_time = epoch_time
            recorder.lr = lr
            print(str(recorder))
            if self.iter % cfg.record_iter == 0:
                recorder.record('train')

            if self.iter % cfg.save_iter == 0:
                net_util.save_model(renderer, optimizer, recorder, self.iter)

            if self.iter % cfg.eval_iter == 0:
                print(colored('Evaluating...', 'yellow'))
                self.val(renderer, self.val_loader, self.evaluator)
                renderer.train()
            scheduler(optimizer, self.iter)
            self.iter += 1

    def val(self, renderer, loader, evaluator):
        torch.cuda.empty_cache()
        renderer.eval()
        for batch in tqdm.tqdm(loader):
            batch = trainer_util.to_cuda(batch, self.device)

            with torch.no_grad():
                out = renderer(batch)
                evaluator.evaluate(out, batch)
        evaluator.summarize()
