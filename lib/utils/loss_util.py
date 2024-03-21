import torch
import torch.nn.functional as F
from lib.config import cfg


def regularize(x, target, p=1):
    crit = (torch.norm(x, dim=-1) - target) ** p
    return crit.mean()


def soft_regularize(x, target, p=1):
    crit = (torch.relu(torch.norm(x, dim=-1) - target)) ** p
    return crit.mean()


def mse(x, y):
    crit = torch.mean((x - y) ** 2)
    return crit


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone()  # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i], :] = rgbs[div_indices[i]:div_indices[i + 1], :]

    return patch_imgs


def lpips_crit(rgbs, patch_masks, targets, div_indices, net):
    rgb = _unpack_imgs(rgbs, patch_masks, torch.tensor(cfg.bg_color).to(rgbs), targets, div_indices)
    targets = targets.to(rgb)
    loss = net(rgb.permute(0, 3, 1, 2), targets.permute(0, 3, 1, 2))
    return torch.mean(loss)


def get_intersection_mask(sdf):
    """
    sdf: n_batch, n_pixel, n_sample
    z_vals: n_batch, n_pixel, n_sample
    """
    sign = torch.sign(sdf[..., :-1] * sdf[..., 1:])
    ind = torch.min(sign * torch.arange(sign.size(2)).flip([0]).to(sign),
                    dim=2)[1]

    inter=sign==-1
    first_inter=torch.cumsum(inter,dim=-1)
    first_inter=first_inter==1
    ind=first_inter*inter
    sign = sign.min(dim=2)[0]
    intersection_mask = sign == -1
    return intersection_mask, ind


def sdf_mask_loss(sdf, occ, iter_step):
    # get pixels that outside the mask or no ray-geometry intersection
    min_sdf = sdf.min(dim=2)[0]
    free_sdf = min_sdf[occ == 0]
    free_label = torch.zeros_like(free_sdf)

    with torch.no_grad():
        intersection_mask, _ = get_intersection_mask(sdf)
    ind = (intersection_mask == False) * (occ == 1)
    sdf = min_sdf[ind]
    label = torch.ones_like(sdf)

    msk_sdf = torch.cat([sdf, free_sdf])
    msk_label = torch.cat([label, free_label])
    # ipdb.set_trace()
    if len(msk_sdf) == 0:
        return torch.tensor(0).to(sdf)
    alpha = 50
    alpha_factor = 2
    alpha_milestones = [10000, 20000, 30000, 40000, 50000]
    for milestone in alpha_milestones:
        if iter_step * torch.cuda.device_count() > milestone:
            alpha = alpha * alpha_factor

    msk_sdf = -alpha * msk_sdf
    mask_loss = F.binary_cross_entropy_with_logits(msk_sdf, msk_label) / alpha
    # if torch.isnan(mask_loss):

    return mask_loss

def alpha_mask_loss(acc_map, occ, iter_step):

    masked = occ == 1
    unmasked = occ == 0
    pred_mask = torch.cat([acc_map[masked], acc_map[unmasked]], dim=-1)
    gt_mask = torch.cat([occ[masked], occ[unmasked]], dim=-1).to(torch.float32)
    mask_loss = mse(pred_mask, gt_mask)
    return mask_loss