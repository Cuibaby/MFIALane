import os
import os.path as osp
import numpy as np
import torchvision
from .base_dataset import BaseDataset
from .registry import DATASETS
import cv2
import torch
import PIL.Image as Image
import logging

@DATASETS.register_module
class VILane(BaseDataset):
    def __init__(self, img_path, data_list, cfg=None):
        super().__init__(img_path, data_list, cfg=cfg)
        self.ori_imgh = 1080
        self.ori_imgw = 1920
       

    def init(self):
        self.logger.info('Loading VIL100 annotations...')
        with open(osp.join(self.list_path, self.data_list)) as f:
            for line in f:
                line_split = line.strip().split(" ")
                
                self.img_name_list.append(line_split[0])
                self.full_img_path_list.append(self.img_path + line_split[0])
                if not self.is_training:
                    continue
                self.label_list.append(self.img_path + line_split[1])
                self.exist_list.append(
                    np.array([int(line_split[2]), int(line_split[3]),
                              int(line_split[4]), int(line_split[5]),
                              int(line_split[6]), int(line_split[7]),
                              int(line_split[8]), int(line_split[9])]))

    def __getitem__(self, idx):
        img = cv2.imread(self.full_img_path_list[idx]).astype(np.float32)
        hw = img.shape[:2]
        img = img[self.cfg.cut_height:, :, :]
        self.ori_imgh = hw[0]
        self.ori_imgw = hw[1]

        sample = {'img': img, 'img_path': self.full_img_path_list[idx],
                'img_name': self.img_name_list[idx], 'hw':hw,}
        
        if self.training:
           temp = Image.open(self.label_list[idx])
           label = np.array(temp)
           if len(label.shape) > 2:
              label = label[:, :, 0]
           
           label = label.squeeze()
           label = label[self.cfg.cut_height:, :]
           sample.update({'mask': label})
           exist = self.exist_list[idx]
           sample.update({'lane_exist': exist})

        sample = self.processes(sample)
        return sample
    def probmap2lane(self, probmaps, exists, h, w, pts=80):
        coords = []
        probmaps = probmaps[1:, ...]
        exists = exists > 0.5
        t = -1
        self.ori_imgh = h - self.cfg.cut_height
        self.ori_imgw = w
        gap = self.ori_imgh / pts
        for probmap, exist in zip(probmaps, exists):
            t += 1
            if exist == 0:
                continue
            
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)
            thr = 0.3
            coordinate = np.zeros(pts)
            cut_height = self.cfg.cut_height
            for i in range(pts):
                y = round(self.cfg.img_height-i*gap/self.ori_imgh*self.cfg.img_height)-1
                if y < 0:
                    break
                line = probmap[y]
                if np.max(line) > thr:
                    coordinate[i] = np.argmax(line)+1
            if np.sum(coordinate > 0) < 4:
                continue
    
            img_coord = np.zeros((pts, 2))
            img_coord[:, :] = -1
            for idx, value in enumerate(coordinate):
                if value > 0:
                    img_coord[idx][0] = round(value*self.ori_imgw/self.cfg.img_width-1)
                    img_coord[idx][1] = round(h - idx*self.ori_imgh/pts-1)
            
            img_coord = img_coord.astype(int)
            coords.append(img_coord)
    
        return coords
