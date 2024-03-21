import torch.nn as nn
import torch
from .embedder import get_embedder


class ColorNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, skip, xyz_multires, view_multires, normal_multires, weight_norm,
                 activation, in_viewdir, in_pts, in_normal):
        super(ColorNetwork, self).__init__()
        self.skip = skip
        if in_pts:
            if xyz_multires > 0:
                self.xyz_embedder, xyz_dim = get_embedder(xyz_multires)
                in_dim = in_dim + xyz_dim
            else:
                in_dim += 3
        dims = [in_dim] + [width for _ in range(depth)] + [out_dim]
        if in_viewdir:
            if view_multires > 0:
                self.view_embedder, view_dim = get_embedder(view_multires)
                dims[0] += view_dim
            else:
                dims[0] += 3
        if in_normal:
            if normal_multires > 0:
                self.normal_embedder, normal_dim = get_embedder(normal_multires)
                dims[0] += normal_dim
            else:
                dims[0] += 3

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            in_dim = dims[l]
            lin = nn.Linear(in_dim, out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise Exception('activation not supported')

    def forward(self, pts, normal, view_dir, feature):
        x = None
        if pts is not None:
            if getattr(self, 'xyz_embedder', False):
                pts = self.xyz_embedder(pts)
            x = pts
        if view_dir is not None:
            if getattr(self, 'view_embedder', False):
                view_dir = self.view_embedder(view_dir)
            if x is None:
                x = view_dir
            else:
                x = torch.cat([pts, view_dir], dim=-1)

        if normal is not None:
            if getattr(self, 'normal_embedder', False):
                normal = self.normal_embedder(normal)
            if x is None:
                x = normal
            else:
                x = torch.cat([x, normal], dim=-1)
        if feature is not None:
            x = torch.cat([x, feature], dim=-1)
        z = x
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip:
                x = torch.cat([x, z], dim=-1)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x
