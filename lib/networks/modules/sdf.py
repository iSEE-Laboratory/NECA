import torch
import torch.nn as nn
import numpy as np
from .embedder import get_embedder
from lib.config import cfg


class SDFNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, skip, xyz_multires,
                 bias, scale, geometric_init, weight_norm, activation, in_pose_feat):
        super(SDFNetwork, self).__init__()

        self.skip = skip

        if xyz_multires > 0:
            self.xyz_embedder, xyz_dim = get_embedder(xyz_multires)
            in_dim = in_dim - 3 + xyz_dim
        if cfg.concat_pose:
            in_dim += cfg.pose_dim
        dims = [in_dim] + [width for _ in range(depth)] + [out_dim]

        self.num_layers = len(dims)

        self.scale = scale

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            if l in self.skip:
                in_dim = dims[l] + dims[0]
            else:
                in_dim = dims[l]
            lin = nn.Linear(in_dim, out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                               np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif xyz_multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif xyz_multires > 0 and l in self.skip:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, xyz, feat=None):
        x = None
        if xyz is not None:
            xyz = xyz * self.scale
            if getattr(self, 'xyz_embedder', False):
                xyz = self.xyz_embedder(xyz)
            x = xyz

        if x is None:
            x = feat
        else:
            if feat is not None:
                x = torch.cat([x, feat], dim=-1)

        z = x
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip:
                x = torch.cat([x, z], dim=-1) / np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x
