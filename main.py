import os
import os.path as osp
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import argparse
import numpy as np
import random
from lanedet.utils.config import Config
from lanedet.engine.runner import Runner 
from lanedet.datasets import build_dataloader


def main():
    
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed
    cfg.flag = True if cfg.haskey('flag') else False
    cfg.work_dirs = args.work_dirs + '/' + cfg.dataset.train.type

    cudnn.benchmark = True
  #  cudnn.fastest = True

    runner = Runner(cfg)

    if args.validate:
        runner.validate()
    else:
        runner.train()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work_dirs', type=str, default='work_dirs',
        help='work dirs')
    parser.add_argument(
        '--load_from', default=None,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--finetune_from', default=None,
        help='whether to finetune from the checkpoint')
    parser.add_argument(
        '--view', action='store_true', 
        help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--gpus', nargs='+', type=int, default='3')
    parser.add_argument('--seed', type=int,
                        default=0, help='random seed')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
