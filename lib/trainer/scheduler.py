import torch
from lib.config import cfg


class ExponentialLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, decay_epochs, gamma=0.1, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.decay_epochs)
                for base_lr in self.base_lrs]


def update_lr(optimizer, iter_step):
    decay_rate = 0.1
    decay_steps = cfg.train.scheduler.decay * 1000
    decay_value = decay_rate ** (iter_step / decay_steps)
    for param_group in optimizer.param_groups:
        base_lr = cfg.train.lr
        new_lrate = base_lr * decay_value
        # else:
        #     new_lrate = cfg.train.lr * decay_value
        param_group['lr'] = new_lrate
