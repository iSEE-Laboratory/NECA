from .yacs import CfgNode as CN
import argparse
import os
import yaml

node = CN()
node.train_light = False


def get_cfg_defaults():
    return node.clone()


def parse_dict(config):
    dic = {}
    for k in config.keys():
        if isinstance(config[k], dict):
            dic[k] = parse_dict(config[k])
        else:
            dic[k] = config[k]
    return dic


def save_cfg_file():
    config = parse_dict(cfg)
    with open(cfg.config_dir, "w") as f:
        yaml.dump(config, f)


def get_path(cfg):
    cfg.log_dir = os.path.join(os.getcwd(), 'result', cfg.exp_name, args.task)
    os.makedirs(cfg.log_dir, exist_ok=True)

    cfg.model_dir = os.path.join(cfg.log_dir, 'model')
    if cfg.task != 'train':
        cfg.model_dir = cfg.model_dir.replace(cfg.task, 'train')

    os.makedirs(cfg.model_dir, exist_ok=True)

    cfg.config_dir = os.path.join(cfg.log_dir, 'config.yaml')
    pt = 'eval'
    if cfg.task != 'train':
        pt = cfg.type
        if cfg.type == 'relight':
            pt += '-' + (cfg.hdr_path[:-4]) + '-frame' + str(cfg.test_dataset.begin_frame)
    if cfg.drop_shadow:
        pt += '/drop-shadow'
    cfg.eval_dir = os.path.join(cfg.log_dir, pt)
    os.makedirs(cfg.eval_dir, exist_ok=True)


def parse_cfg(cfg, args):
    cfg.resume = args.resume
    cfg.devices = args.devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.devices
    cfg.distributed = len(args.devices) > 1
    cfg.task = args.task
    cfg.type = args.type
    get_path(cfg)


def make_cfg(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(os.getcwd(), 'config/default.yaml'))
    cfg.merge_from_file(args.cfg)
    cfg.local_rank = 0
    if args.task == 'evaluate':
        if args.type == 'novel_view':
            cfg.merge_from_other_cfg(cfg.novel_view_eval_cfg)
        elif args.type == 'novel_pose':
            cfg.merge_from_other_cfg(cfg.novel_pose_eval_cfg)
    elif args.task == 'visualize':
        if args.type == 'relight':
            cfg.merge_from_other_cfg(cfg.relight_cfg)
        elif args.type == 'reshape':
            cfg.merge_from_other_cfg(cfg.reshape_cfg)
        elif args.type == 'recloth':
            cfg.merge_from_other_cfg(cfg.recloth_cfg)
        elif args.type == 'reshadow':
            cfg.merge_from_other_cfg(cfg.reshadow_cfg)

    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="config/zju_mocap/313.yaml", type=str)
parser.add_argument("--devices", default="0", type=str, help="cuda visible device")
parser.add_argument("--task", default="train", type=str)
parser.add_argument("--type", default='novel_view', type=str)
parser.add_argument("--resume", action='store_true', default=False, help='resume training')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()

cfg = make_cfg(args)
