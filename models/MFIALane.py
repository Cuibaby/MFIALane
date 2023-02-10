import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from models.registry import NET
from .resnet import ResNetWrapper 
from .decoder import BUSD, PlainDecoder 

class ASPPNet(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPPNet, self).__init__()
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

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

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
        conv_stride = cfg.mfia.conv_stride
        
        dilations = [1,2,4,8]
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        for i in range(self.iter):
            conv_vert1 = ASPPNet(chan, chan, dilation=dilations[0])
            conv_vert2 = ASPPNet(chan, chan, dilation=dilations[1])

            setattr(self, 'conv_d'+str(i), conv_vert1)
            setattr(self, 'conv_u'+str(i), conv_vert2)

            conv_hori1 = ASPPNet(chan, chan, dilation=dilations[2])
            conv_hori2 = ASPPNet(chan, chan, dilation=dilations[3])
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
        self.conv1x1 = nn.Conv2d(128, cfg.num_classes, kernel_size=1, stride=1, bias=False)
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
        self.fc = nn.Linear(128, cfg.num_classes-1)
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
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
    def forward(self, batch):
        feat = self.backbone(batch)
        feat = self.mfia(feat)
        avg_feat = self.global_avg_pool(feat) 
        feat = feat * avg_feat
        seg = self.decoder(feat)
        exist = self.heads(avg_feat)
        output = {'seg': seg, 'exist': exist}
        return output
