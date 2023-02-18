import time
import torch
import numpy as np
from tqdm import tqdm
import pytorch_warmup as warmup
import argparse
import os
import torchvision
from utils.config import Config
from models.registry import build_net
from runner.registry import build_trainer, build_evaluator
from runner.optimizer import build_optimizer
from runner.scheduler import build_scheduler
from datasets import build_dataloader
from runner.recorder import build_recorder
from runner.net_utils import save_model, load_network, load_network_specified
from mmcv.cnn.utils import get_model_complexity_info
def arg_set():
    parser = argparse.ArgumentParser(description='PyTorch MFIALane')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--gpus', type=int, default=0, nargs='+',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[288, 800],
        help='input image size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = arg_set()
    cfg = Config.fromfile(args.config)
    torch.backends.cudnn.benchmark = True
   
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    use_cuda = torch.cuda.is_available()
    
    device = torch.device("cuda" if use_cuda else "cpu")
    #构造列表中的次序转换
   
    gpus_id = [i for i in range(len(args.gpus))]
   
    net = build_net(cfg)
    net = torch.nn.DataParallel(net,device_ids=gpus_id).cuda()
    net.eval()
    
    x = torch.zeros((1,3,cfg.img_height,cfg.img_width)).to(device) + 1

    t_all = []
    for i in range(100):
        y = net(x)
    for i in range(100):
       t1 = time.time()
       y = net(x)
       t2 = time.time()
       t_all.append(t2 - t1)

    print('average time:', np.mean(t_all) / 1)
    print('average fps:',1 / np.mean(t_all))

    print('fastest time:', min(t_all) / 1)
    print('fastest fps:',1 / min(t_all))

    print('slowest time:', max(t_all) / 1)
    print('slowest fps:',1 / max(t_all))

