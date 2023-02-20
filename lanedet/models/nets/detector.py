import torch.nn as nn
import torch

from lanedet.models.registry import NETS
from ..registry import build_backbones, build_aggregator, build_heads, build_necks


@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))) if cfg.haskey('ca') else None
    def get_lanes(self, output):
        return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}
        fea = self.backbone(batch['img'])

        if self.aggregator:
            
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)
        y = None
        if self.global_avg_pool is not None:
           y = self.global_avg_pool(fea[-1])
           fea[-1] = y * fea[-1]
        if self.training:  
            
            out = self.heads(fea, y, batch=batch)
            output.update(self.heads.loss(out, batch))
        else:
            output = self.heads(fea, y)

        return output
