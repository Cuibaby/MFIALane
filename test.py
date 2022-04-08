
import torch

pretrained_net = torch.load('weights/culane_resnet50.pth')['net']
for k, v in pretrained_net.items():
    print(k,v.size())
