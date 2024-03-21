import torch.nn as nn
import torch


class BetaNetwork(nn.Module):
    def __init__(self,init_val):
        super(BetaNetwork, self).__init__()
        self.register_parameter('beta', nn.Parameter(torch.tensor(init_val)))

    def forward(self):
        beta = self.beta
        return beta
