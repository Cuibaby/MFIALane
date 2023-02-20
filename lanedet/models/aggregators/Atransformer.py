import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lanedet.models.registry import AGGREGATORS 
 

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        
        super(HAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
       
        self.proj_drop = nn.Dropout(proj_drop)
       

    def forward(self, x):
       
        B, C, H, W = x.shape
        x = x.permute(0,3,2,1).reshape(-1, H, C)
        qkv = self.qkv(x).reshape(B, W, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, W, n_head, H, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, W, n_head, H, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, W, n_head, H, H
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, W, n_head, H, dim -> (t(2,3)) B, W, H, n_head, head_dim ->reshape B, W, H, C
        attn = (attn @ v).transpose(2, 3).reshape(B, W, H, C)
        
        x = attn.permute(0, 3, 2, 1)
        x = self.proj_drop(x)
       
        return x

class WAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        
        super(WAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
       
        self.proj_drop = nn.Dropout(proj_drop)
       

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).reshape(-1, W, C)
        qkv = self.qkv(x).reshape(B, H, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, H, n_head, W, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, n_head, W, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, n_head, W, W
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, H, n_head, W, dim -> (t(2,3)) B, H, W, n_head, head_dim ->reshape B, H, W, C
        attn = (attn @ v).transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = attn
        x = self.proj_drop(x)
        return x
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(2 * dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: B, C, H, W
        """
        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2 
        x = self.reduction(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Attention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class GroupBlock(nn.Module):
    def __init__(self, dim, num_heads, channel, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(GroupBlock, self).__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attnh = HAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        
        self.attnw = WAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm11 = norm_layer(channel)
        self.norm12 = norm_layer(channel)
        self.norm21 = norm_layer(channel)
        self.norm22 = norm_layer(channel)
    def forward(self, x):
         
        x = x + self.attnh(x)
        x = x.permute(0,2,3,1)
        x = self.norm11(x)
        x = x + self.mlp1(x)
        x = self.norm12(x).permute(0,3,1,2)
        x = x + self.attnw(x)
        x = x.permute(0,2,3,1)
        x = self.norm21(x)
        x = x + self.mlp2(x)
        x = self.norm22(x).permute(0,3,1,2)
      
        return x


@AGGREGATORS.register_module
class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads, input_channel, cfg):
        super(Transformer, self).__init__()
        self.cfg = cfg
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.input_channel = input_channel
        self.reduction = PatchMerging(dim = self.dim)
        self.blocks = nn.ModuleList()
        for i in range(self.depth-4):
             self.blocks.append(GroupBlock(dim = self.dim , num_heads = self.num_heads, channel = self.input_channel))
        for i in range(4):
             self.blocks.append(GroupBlock(dim = self.dim*2, num_heads = self.num_heads, channel = self.input_channel*2))

    def forward(self, fea):
        
        for i in range(self.depth):
            if i == 2:
               fea = self.reduction(fea)
            fea = self.blocks[i](fea)
            
        
        return fea

