import imp
from . import ExponentialLR, Recorder, update_lr
import torch
import torch.utils.data
import numpy as np
import cv2
import time

def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    network = imp.load_source(module, path).Network()
    return network


def make_renderer(cfg, network):
    module = cfg.renderer_module
    path = cfg.renderer_path
    renderer = imp.load_source(module, path).Renderer(network)
    return renderer

def make_optimizer(cfg, net):
    params = []

    weight_decay = cfg.train.weight_decay
    # import ipdb;ipdb.set_trace()
    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        module = key.split('.')[0]
        lr = cfg.train.lr
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' == cfg.train.optim:
        optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay)
    else:
        raise Exception('optimizer is not supported')

    return optimizer


def make_lr_scheduler(cfg, optimizer):

    return update_lr


def make_recorder(cfg):
    return Recorder(cfg)


def make_evaluator(cfg, lpips):
    module = cfg.evaluator_module
    path = cfg.evaluator_path
    evaluator = imp.load_source(module, path).Evaluator(lpips)
    return evaluator


def make_visualizer(cfg):
    module = cfg.visualizer_module
    path = cfg.visualizer_path
    visualizer = imp.load_source(module, path).Visualizer()
    return visualizer


def make_dataset(cfg, is_train):

    if is_train:
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
        args = cfg.train_dataset
    else:
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
        args = cfg.test_dataset
    dataset = imp.load_source(module, path).Dataset(**args)

    return dataset


def make_sampler(dataset, shuffle, distributed, is_train):
    if distributed and is_train:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def worker_init_fn(worker_id):
    cv2.setNumThreads(1)  # MARK: OpenCV undistort is why all cores are taken
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2 ** 16))))


def make_dataloader(cfg, is_train=True):
    if is_train:
        bs = cfg.train.batch_size
        shuffle = cfg.train.shuffle
    else:
        bs = cfg.test.batch_size
        shuffle = cfg.test.shuffle
    dataset = make_dataset(cfg, is_train)
    sampler = make_sampler(dataset, shuffle, cfg.distributed, is_train)
    batch_sampler = torch.utils.data.BatchSampler(sampler, bs, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=cfg.num_workers)
    return data_loader
