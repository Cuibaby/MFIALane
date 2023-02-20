import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import math
from lanedet.models.registry import AGGREGATORS 
from .aspp import ASPP



class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
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
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pointwise(x) 
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
class cbam(nn.Module):
     def __init__(self, planes):
        super(cbam,self).__init__()
        self.ca = ChannelAttention(in_planes = planes)# planesÊÇfeature mapµÄÍ¨µÀ¸öÊý
        self.sa = SpatialAttention()
     def forward(self, x):
        x = self.ca(x) * x  # ¹ã²¥»úÖÆ
        x = self.sa(x) * x  # ¹ã²¥»úÖÆ
        return x
@AGGREGATORS.register_module
class MFIA(nn.Module):
    def __init__(self, direction, alpha, iter, conv_stride, cfg):
        super(MFIA, self).__init__()
        self.cfg = cfg
        self.iter = iter
        chan = cfg.featuremap_out_channel 
        fea_stride = cfg.featuremap_out_stride
        self.height = cfg.img_height // fea_stride
        self.width = cfg.img_width // fea_stride
        self.alpha = alpha
        self.direction = direction
        conv_stride = conv_stride
        
        dilations = [1,2,4,8]
       
        for i in range(self.iter):
            conv_vert1 = ASPP_module(chan, chan, dilation=dilations[0])
            conv_vert2 = ASPP_module(chan, chan, dilation=dilations[1])

            setattr(self, 'conv_d'+str(i), conv_vert1)
            setattr(self, 'conv_u'+str(i), conv_vert2)

            conv_hori1 = ASPP_module(chan, chan, dilation=dilations[2])
            conv_hori2 = ASPP_module(chan, chan, dilation=dilations[3])
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
      
        for direction in self.direction:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                if direction in ['d', 'u']:
                   x.add_(self.alpha * F.relu(conv(x[..., idx, :])))
                else:
                   x.add_(self.alpha * F.relu(conv(x[..., idx])))
                
               
        return x



