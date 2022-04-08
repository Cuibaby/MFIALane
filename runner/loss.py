
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChannelWiseDivergence(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=4.0,
        loss_weight=1.0,
    ):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        """Forward computation.
        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, W, H).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, W, H).
        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, W, H = preds_S.shape

        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = self.loss_weight * loss / (C * N)

        return loss

class KDLoss(nn.Module):
    """Knowledge Distillation Loss"""

    def __init__(self, T = 4.0):
        super().__init__()
        self.t = T

    def forward(self, stu_pred, tea_pred):
        s = F.log_softmax(stu_pred / self.t, dim=1)
        t = F.softmax(tea_pred / self.t, dim=1)
        loss = F.kl_div(s, t, size_average=False) * (self.t**2) / stu_pred.shape[0]
        return loss

class SKDLoss(nn.Module):
    def __init__(self):
        super(SKDLoss, self).__init__()
        
        self.mseloss = nn.MSELoss()
    def forward(self, x1, x2):
        loss = 0.
        x2 = x2.detach()
        loss += self.mseloss(x1,x2)
        mean_x1 = x1.mean(dim=(2, 3), keepdim=False)
        mean_x2 = x2.mean(dim=(2, 3), keepdim=False)
        std_x1 = x1.std(dim=(2, 3), keepdim=False)
        std_x2 = x2.std(dim=(2, 3), keepdim=False)
        loss += torch.mean(torch.pow(mean_x1 - mean_x2, 2)) + torch.mean(torch.pow(std_x1 - std_x2, 2))
        return loss

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        #og_softmaxÊÇÖ¸ÔÚsoftmaxº¯ÊýµÄ»ù´¡ÉÏ£¬ÔÙ½øÐÐÒ»´ÎlogÔËËãµÃres[n,m]£¬´ËÊ±½á¹ûÓÐÕýÓÐ¸º£¬
        # ground_truth=[n]nn.NLLLoss´ËÊ±Îª,°´ÕÕÎ»ÖÃÈ¥µô¸ººÅÏà¼ÓÆ½¾ùÖµ×÷ÎªËðÊ§
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()
    def forward(self,logits):
        n,c,h,w = logits.shape
        loss_all = []
        logits = torch.softmax(logits,dim=1)
        for i in range(0,h-1):
            loss_all.append(torch.abs(logits[:,1:,i,:] - logits[:,1:,i+1,:]))
        #loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss,torch.zeros_like(loss))
class GemRelationLoss(nn.Module):
    def __init__(self):
        super(GemRelationLoss, self).__init__()
    def forward(self,logits, params):
        n,c,h,w = logits.shape
        logits = torch.softmax(logits,dim=-1)
        loss_all = []
        embedding = torch.Tensor(np.arange(w)).float().to(logits.device).view(1,1,1,-1)
        pos = torch.sum(logits*embedding,dim = -1) 
        for i in range(n):
            for j in range(len(params)):
                for z in range(params[i][j][-3],params[i][j][-2]):
                    loss_all.append(pos[i,params[i][j][-1],z] - pos[i,params[i][j][-1],z+1])
        #loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss,torch.zeros_like(loss))

class GemLoss(nn.Module):
    def __init__(self):
        super(GemLoss, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
    def forward(self, x):
        n,dim,num_rows,num_cols = x.shape
        x = torch.nn.functional.softmax(x,dim=-1) #x[:,1:,]
        embedding = torch.Tensor(np.arange(num_cols)).float().to(x.device).view(1,1,1,-1)
        pos = torch.sum(x[:,1:,]*embedding,dim = -1) 
       
        diff_list1 = [] 
        for i in range(0,num_rows - 1):
            diff_list1.append(torch.abs(pos[:,:,i] - pos[:,:,i+1]))

        loss = 0
        for i in range(len(diff_list1)-1):
            loss += self.l2(diff_list1[i],diff_list1[i+1])
        loss /= len(diff_list1) - 1
        return loss

class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
    def forward(self, x):
        n,dim,num_rows,num_cols = x.shape
        x = torch.nn.functional.softmax(x[:,:dim-1,:,:],dim=1)
        embedding = torch.Tensor(np.arange(dim-1)).float().to(x.device).view(1,-1,1,1)
        pos = torch.sum(x*embedding,dim = 1) 
        diff_list1 = [] 
        for i in range(0,num_rows // 2):
            diff_list1.append(pos[:,i,:] - pos[:,i+1,:])

        loss = 0
        for i in range(len(diff_list1)-1):
            loss += self.l2(diff_list1[i],diff_list1[i+1])
        loss /= len(diff_list1) - 1
        return loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()
class Dice(nn.Module):
    def __init__(self):
        super(Dice,self).__init__()
        self.alp = 1e-8
    def forward(self,pred,gt):
        loss = 1 - (2 * (pred * gt).sum() + self.alp) / (torch.pow(pred,2).sum() + torch.pow(gt,2).sum() + self.alp)
        return loss

def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return (1-d).mean()



