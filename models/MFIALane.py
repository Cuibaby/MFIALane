import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from models.registry import NET
from .resnet import ResNetWrapper 
from .decoder import BUSD, PlainDecoder 

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation, pointwise=False):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation #groups = inplanes,
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,groups = inplanes,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes, eps=1e-03)
        self.relu = nn.ReLU()
       
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=False) if pointwise else None
       
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pointwise is not None:
            x = self.pointwise(x) 
            x = self.relu(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class MFIA(nn.Module):
    def __init__(self, cfg):
        super(MFIA, self).__init__()
        self.iter = cfg.mfia.iter
        chan = cfg.mfia.input_channel
        fea_stride = cfg.backbone.fea_stride
        self.height = cfg.img_height // fea_stride
        self.width = cfg.img_width // fea_stride
        self.alpha = cfg.mfia.alpha
        pointwise = cfg.mfia.pointwise
        dilations = [1,2,4,8]
        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        for i in range(self.iter):
            conv_vert1 = ASPP_module(chan, chan, dilation=dilations[0], pointwise=pointwise)
            conv_vert2 = ASPP_module(chan, chan, dilation=dilations[1], pointwise=pointwise)

            setattr(self, 'conv_d'+str(i), conv_vert1)
            setattr(self, 'conv_u'+str(i), conv_vert2)

            conv_hori1 = ASPP_module(chan, chan, dilation=dilations[2], pointwise=pointwise)
            conv_hori2 = ASPP_module(chan, chan, dilation=dilations[3], pointwise=pointwise)
            setattr(self, 'conv_r'+str(i), conv_hori1)
            setattr(self, 'conv_l'+str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height //
                     2**(self.iter - i)) % self.height
            setattr(self, 'idx_d'+str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height //
                     2**(self.iter - i)) % self.height
            setattr(self, 'idx_u'+str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //
                     2**(self.iter - i)) % self.width
            setattr(self, 'idx_r'+str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //
                     2**(self.iter - i)) % self.width
            setattr(self, 'idx_l'+str(i), idx_l)

    def forward(self, x):
        x = x.clone()

        for direction in ['d','u']: # 'u'
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx, :])))
               

        for direction in ['r','l']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx])))
               
        return x

class DetailHead(nn.Module):
    def __init__(self,cfg=None):
        super(DetailHead, self).__init__()
        self.cfg = cfg
        self.conv1x1 = nn.Conv2d(self.cfg.mfia.input_channel, cfg.num_classes, kernel_size=1, stride=1, bias=False)
    def forward(self, x):
        x = F.interpolate(x,size=[self.cfg.img_height,  self.cfg.img_width],
                           mode='bilinear', align_corners=False)
        x = self.conv1x1(x) 
        return x

class ExistHead(nn.Module):
    def __init__(self, cfg=None):
        super(ExistHead, self).__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(0.1)  
        self.fc = nn.Linear(self.cfg.mfia.input_channel, cfg.num_classes-1)
    def forward(self, x):
        x = self.dropout(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

@NET.register_module
class MFIALane(nn.Module):
    def __init__(self, cfg):
        super(MFIALane, self).__init__()
        self.cfg = cfg
        self.backbone = ResNetWrapper(cfg)
        self.mfia = MFIA(cfg)
        self.decoder = eval(cfg.decoder)(cfg)
        self.heads = ExistHead(cfg) 
        # self.ca = False if 'CULane' not in cfg.dataset_path else True
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
    def forward(self, batch):
        fea = self.backbone(batch)
        fea = self.mfia(fea)
        avg_fea = self.global_avg_pool(fea) 
        fea = fea * avg_fea
        seg = self.decoder(fea)
        exist = self.heads(avg_fea)
        output = {'seg': seg, 'exist': exist}
        return output
