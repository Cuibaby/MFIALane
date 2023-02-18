import torch.nn as nn
import torch
import torch.nn.functional as F
from runner.logger import get_logger
import logging
from runner.registry import EVALUATOR 
import json
import os
import subprocess
from shutil import rmtree
import cv2
import numpy as np

def check():
    import subprocess
    import sys
    FNULL = open(os.devnull, 'w')
    result = subprocess.call(
        './runner/evaluator/culane/lane_evaluation/evaluate', stdout=FNULL, stderr=FNULL)
    if result > 1:
        print('There is something wrong with evaluate tool, please compile it.')
        sys.exit()

def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k : v for k,v in zip(keys,values)}
    return res

def call_culane_eval(data_dir, output_path='./output'):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir=os.path.join(output_path, 'lines')+'/'

    w_lane=30
    iou=0.5;  # Set iou to 0.3 or 0.5
    im_w=1640
    im_h=590
    frame=1
    # print('The IoU threshold is:', iou)
    list_dir = os.listdir(os.path.join(data_dir,'test/list'))
    res_all = {}
    if not os.path.exists(os.path.join(output_path,'txt')):
        os.mkdir(os.path.join(output_path,'txt'))
    for i in list_dir:
        with open(os.path.join(data_dir,'test/list/'+i),'r') as fp:
            lines = fp.readlines()[0][:-1]
        fp.close()
        img = cv2.imread(os.path.join(data_dir,lines))
        im_h,im_w = img.shape[:2]
        list0 = os.path.join(data_dir,'test/list/'+i)
        out0 = os.path.join(output_path,'txt',i)
        eval_cmd = './runner/evaluator/culane/lane_evaluation/evaluate'
        os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
        res_all[i] = read_helper(out0)
    return res_all

@EVALUATOR.register_module
class VILane(nn.Module):
    def __init__(self, cfg):
        super(VILane, self).__init__()
        # Firstly, check the evaluation tool
        check()
        self.cfg = cfg 
        self.blur = torch.nn.Conv2d(
            9, 9, 9, padding=4, bias=False, groups=9).cuda()
        torch.nn.init.constant_(self.blur.weight, 1 / 81)
        self.logger = logging.getLogger(__name__)
        self.out_dir = os.path.join(self.cfg.work_dir, 'lines')
        if cfg.view:
            self.view_dir = os.path.join(self.cfg.work_dir, 'vis')

    def evaluate(self, dataset, output, batch):
        seg, exists = output['seg'], output['exist']
        predictmaps = F.softmax(seg, dim=1).cpu().numpy()
        exists = exists.cpu().numpy()
        batch_size = seg.size(0)
        hw = batch['meta']['hw']
        img_name = batch['meta']['img_name']
        img_path = batch['meta']['full_img_path']
        for i in range(batch_size):
            h,w = hw[0][i].item(),hw[1][i].item()
            coords = dataset.probmap2lane(predictmaps[i], exists[i], h, w)
            outname = self.out_dir + img_name[i][:-4] + '.lines.txt'
            outdir = os.path.dirname(outname)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            f = open(outname, 'w')
            for coord in coords:
                xx = []
                yy = []
                for x, y in coord:
                    if x < 0 and y < 0:
                        continue
                #     xx.append(x)
                #     yy.append(y)

                # params = np.polyfit(np.array(yy), np.array(xx), 2)
                # yy = np.array(yy)
                # xx = params[0] * yy * yy + params[1] * yy + params[2]
                    
                # for index,(x,y) in enumerate(zip(xx,yy)):
                
                    f.write('%d %d ' % (x, y))
                f.write('\n')
            f.close()

            if self.cfg.view:
                img = cv2.imread(img_path[i]).astype(np.float32)
                dataset.view(img, coords, self.view_dir+img_name[i])


    def summarize(self):
        self.logger.info('summarize result...')
        eval_list_path = os.path.join(
            self.cfg.dataset_path, "list", self.cfg.dataset.val.data_list)
        #prob2lines(self.prob_dir, self.out_dir, eval_list_path, self.cfg)
        res = call_culane_eval(self.cfg.dataset_path, output_path=self.cfg.work_dir)
        TP,FP,FN = 0,0,0
        out_str = 'Copypaste: '
        for k, v in res.items():
            val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
            val_tp, val_fp, val_fn = int(v['tp']), int(v['fp']), int(v['fn'])
            val_p, val_r, val_f1 = float(v['precision']), float(v['recall']), float(v['Fmeasure'])
            TP += val_tp
            FP += val_fp
            FN += val_fn
            self.logger.info(k + ': ' + str(v))
            out_str += k
            for metric, value in v.items():
                out_str += ' ' + str(value).rstrip('\n')
            out_str += ' '
        P = TP * 1.0 / (TP + FP + 1e-9)
        R = TP * 1.0 / (TP + FN + 1e-9)
        F = 2*P*R/(P + R + 1e-9)
        overall_result_str = ('Overall Precision: %f Recall: %f F1: %f' % (P, R, F))
        self.logger.info(overall_result_str)
        out_str = out_str + overall_result_str
        self.logger.info(out_str)

        # delete the tmp output
    #    rmtree(self.out_dir)
        return F
