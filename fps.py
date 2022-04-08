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
    parser = argparse.ArgumentParser(description='PyTorch UNet-ConvLSTM')
    parser.add_argument('--model',type=str, default='UNet',help='( UNet-ConvGRU | UNet | ')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.008, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gpus', type=int, default=0, nargs='+',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='whether use last state')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = arg_set()
    cfg = Config.fromfile(args.config)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
   
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    use_cuda = torch.cuda.is_available()
    
    device = torch.device("cuda" if use_cuda else "cpu")
    #构造列表中的次序转换
   
    gpus_id = [i for i in range(len(args.gpus))]
   
    net = build_net(cfg)
    net = torch.nn.DataParallel(net,device_ids=gpus_id).cuda()
    net.eval()
    
    x = torch.zeros((1,3,384,640)).to(device) + 1

    t_all = []
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

