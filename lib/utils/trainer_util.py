import torch

def to_cuda(batch, device):
    for v in batch:
        if not isinstance(batch[v], torch.Tensor) and not isinstance(batch[v], dict):
            batch[v] = torch.from_numpy(batch[v])
        if isinstance(batch[v], dict):
            for k in batch[v]:
                batch[v][k] = batch[v][k].to(device)
        else:
            batch[v] = batch[v].to(device)
    return batch


def add_iter_step(batch, iter_step):
    batch['iter_step'] = iter_step


def reduce_loss_stats(loss_stats):
    reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
    return reduced_losses
