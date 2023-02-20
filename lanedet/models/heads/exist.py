import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from ..registry import HEADS 


@HEADS.register_module
class Exist(nn.Module):
    def __init__(self, cfg=None):
        super(Exist, self).__init__()
        self.cfg = cfg
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.dropout = nn.Dropout2d(0.1)  
        self.fc = nn.Linear(cfg.featuremap_out_channel, cfg.num_classes-1)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = self.dropout(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        output = {'exist': x}
        return output
