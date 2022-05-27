import time
import torch
import numpy as np
from tqdm import tqdm
import pytorch_warmup as warmup
import shutil
from models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from datasets import build_dataloader
from .recorder import build_recorder
from .net_utils import save_model, load_network, load_network_specified
from .VIL import write_mask
from mmcv.cnn.utils import get_model_complexity_info
class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('the model parameter is :', parameters / 1e6)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.evaluator = build_evaluator(self.cfg)
        self.warmup_scheduler = warmup.LinearWarmup(
            self.optimizer, warmup_period=5000)
        self.metric = 0.75
        if self.cfg.dataset['train']['type'] == 'TuSimple':
            self.metric = 0.9654
        if self.cfg.dataset.train.type == 'VILane':
            self.metric = 0.84
        
    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network_specified(self.net, self.cfg.load_from,
                 logger=self.recorder.logger)
       # load_network(self.net, self.cfg.load_from,
       #         finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            batch[k] = batch[k].cuda()
        return batch
    
    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.trainer.forward(epoch, self.net, data)
            self.optimizer.zero_grad()
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.warmup_scheduler.dampen()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('start training...')
        self.trainer = build_trainer(self.cfg)
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)
        val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)

        for epoch in range(self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate(val_loader)
               # self.save_ckpt()
            elif self.recorder.step >= self.cfg.total_iter:
                self.save_ckpt()
                break

    def validate(self, val_loader):
        self.net.eval()
        for i, data in enumerate(tqdm(val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data['img'])
                
                self.evaluator.evaluate(val_loader.dataset, output, data)

        metric = self.evaluator.summarize()
        if not metric:
            return
        if metric > self.metric:
            self.metric = metric
            self.save_ckpt(is_best=True)
        self.recorder.logger.info('Best metric: ' + str(self.metric))

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler,
                self.recorder, is_best)
