from lpips import LPIPS
from lib.trainer import Trainer
from lib.config import cfg, args, save_cfg_file
from lib.trainer.loader import *
from lib.utils.net_util import load_model, load_net, set_requires_grad
import lib.utils.trainer_util as trainer_util
import torch
import torch.utils.data
from lib.datasets.zju_mocap_dataset import Dataset as TrainDataset
import numpy as np
import random
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataloader import DataLoader
import tqdm
from lib.utils.trainer_util import to_cuda
import lib.utils.net_util as net_util

torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def train(rank, world_size):
    if cfg.distributed:
        cfg.local_rank = rank
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
        synchronize()

    save_cfg_file()

    train_dataset = TrainDataset(**cfg.train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
                              num_workers=cfg.num_workers)

    device = torch.device('cuda:{}'.format(cfg.local_rank))
    network = make_network(cfg).to(device)
    # print(sum(p.numel() for p in network.parameters()) / 1000000.0)
    renderer = make_renderer(cfg, network)
    lpips = LPIPS(net='vgg').to(device)
    set_requires_grad(lpips, False)

    trainer = Trainer(device, lpips)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    if cfg.local_rank == 0:
        evaluator = make_evaluator(cfg, lpips)
        val_dataset = TrainDataset(**cfg.test_dataset)
        val_loader = DataLoader(val_dataset, batch_size=cfg.test.batch_size, shuffle=cfg.test.shuffle,
                                num_workers=cfg.num_workers)

        trainer.val_loader = val_loader
        trainer.evaluator = evaluator
    start_iter = load_model(renderer, optimizer, recorder, cfg.resume)
    trainer.iter = start_iter + 1
    if cfg.distributed:
        renderer = DDP(renderer, device_ids=[cfg.local_rank])

    epoch = start_iter // len(train_loader) + 1
    # for epoch in range(start_iter + 1, cfg.train.max_iter + 1):
    while True:
        if trainer.iter > cfg.train.max_iter:
            break
        recorder.epoch = epoch
        if trainer.iter >= cfg.stop_lpips_iter:

            net_util.save_model(renderer, optimizer, recorder, trainer.iter, name=str(cfg.stop_lpips_iter))
            train_dataset.update_ray_mode('image')
            cfg.ray_mode = 'image'
        elif trainer.iter >= cfg.lpips_iter:
            net_util.save_model(renderer, optimizer, recorder, trainer.iter, name=str(cfg.lpips_iter))
            train_dataset.update_ray_mode('patch')
            cfg.ray_mode = 'patch'

        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        trainer.train(epoch, train_loader, renderer, optimizer, scheduler, recorder)
        epoch += 1

def run_train():
    if cfg.distributed:
        world_size = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '6449'
        mp.spawn(train,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    else:
        train(0, 0)


def run_evaluate():
    device = torch.device('cuda:{}'.format(cfg.local_rank))
    network = make_network(cfg).to(device)
    renderer = make_renderer(cfg, network)
    load_net(renderer)
    lpips = LPIPS(net='vgg').to(device)
    set_requires_grad(lpips, False)
    evaluator = make_evaluator(cfg, lpips)
    val_loader = make_dataloader(cfg, False)
    renderer.eval()
    for batch in tqdm.tqdm(val_loader):
        batch = to_cuda(batch, device)
        trainer_util.add_iter_step(batch, 100000)
        with torch.no_grad():
            out = renderer(batch)
            evaluator.evaluate(out, batch)
    evaluator.summarize()


def run_visualize():
    device = torch.device('cuda:{}'.format(cfg.local_rank))

    if cfg.type=='recloth' or cfg.type=='reshadow':
        from lib.renderers.recloth import Renderer

        network=make_network(cfg).to(device)
        ori_key_pose_num=cfg.key_pose_num
        cfg.key_pose_num=cfg.upper_key_pose_num
        upper=make_network(cfg).to(device)
        cfg.key_pose_num = cfg.lower_key_pose_num
        lower=make_network(cfg).to(device)
        cfg.key_pose_num=ori_key_pose_num
        model_dir = os.path.join(cfg.model_dir, 'latest.pth')
        model=torch.load(model_dir,'cpu')['net']
        state_dict={}

        for k,v in model.items():
            state_dict[k[8:]]=v
        network.load_state_dict(state_dict)

        model = torch.load(model_dir.replace(cfg.exp_name,cfg.upper_name), 'cpu')['net']
        state_dict = {}
        for k, v in model.items():
            state_dict[k[8:]] = v
        upper.load_state_dict(state_dict)

        model = torch.load(model_dir.replace(cfg.exp_name, cfg.lower_name), 'cpu')['net']
        state_dict = {}
        for k, v in model.items():
            state_dict[k[8:]] = v
        lower.load_state_dict(state_dict)
        renderer=Renderer(network,upper,lower)
    else:
        network = make_network(cfg).to(device)
        renderer = make_renderer(cfg, network)
        load_net(renderer)

    val_loader = make_dataloader(cfg, False)
    visualizer = make_visualizer(cfg)
    renderer.eval()
    for batch in tqdm.tqdm(val_loader):
        batch = to_cuda(batch, device)
        with torch.no_grad():
            out = renderer(batch)
            visualizer.visualize(out, batch)

if __name__ == '__main__':
    set_seed(cfg.local_rank + 3407)
    globals()['run_' + args.task]()
