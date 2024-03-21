import torch.nn as nn
import torch
from lib.config import cfg


class Renderer(nn.Module):
    def __init__(self, network):
        super(Renderer, self).__init__()
        self.network = network

    def forward(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = cfg.chunk
        all_ret = {}

        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            ret = self.network(ray_o_chunk, ray_d_chunk, near_chunk, far_chunk, batch)
            if not self.training:
                ret = {k: ret[k].detach().cpu() for k in ret.keys()}

            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        
        ret = {}

        for k in all_ret:
            if k in ['tv_loss']:
                ret[k] = all_ret[k][0]
            else:
                ret[k] = torch.cat(all_ret[k], 1)

        return ret
