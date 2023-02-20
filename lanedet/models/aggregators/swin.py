
import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import math
from lanedet.models.registry import AGGREGATORS 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

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
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = img_size
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).permute(0,2,3,1)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x.permute(0,3,1,2)
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x.permute(0,3,1,2)

class HAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, H, C, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        
        super(HAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
#        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, H, C)).cuda()
#        trunc_normal_(self.absolute_pos_embed, std=.02)
       
   
    def forward(self, x):
       
        B, C, H, W = x.shape
        x = x.permute(0,3,2,1).reshape(-1, H, C)
      #  x = x + self.absolute_pos_embed
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
    def __init__(self, dim, W, C, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        
        super(WAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
       
        self.proj_drop = nn.Dropout(proj_drop)
 #       self.absolute_pos_embed = nn.Parameter(torch.zeros(1, W, C)).cuda()
 #       trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).reshape(-1, W, C)
 #       x = x + self.absolute_pos_embed 
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
    def __init__(self, dim, num_heads, channel, H, W, C, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(GroupBlock, self).__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attnh = HAttention(dim, H, C, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        
        self.attnw = WAttention(dim, W, C, num_heads, qkv_bias, qk_scale, attn_drop, drop)
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


class BasicLayer(nn.Module):

    def __init__(self, dim, num_heads, depth, downsample, H, W, C):

        super().__init__()
        self.dim = dim
        self.depth = depth
        # build blocks
        self.blocks = nn.ModuleList([GroupBlock(dim = dim, num_heads = num_heads, channel = dim, H = H, W = W, C = C)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = PatchMerging(dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class ExistHead(nn.Module):
    def __init__(self, cfg=None):
        super(ExistHead, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2d(0.1)  
        self.fc = nn.Linear(self.cfg.out_channel, cfg.num_classes-1)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

@AGGREGATORS.register_module
class SWIN(nn.Module):
    def __init__(self, cfg):
        super(SWIN, self).__init__()
        self.cfg = cfg
        self.num_heads = self.cfg.num_heads
        self.depth = self.cfg.depth
        self.embed_dim = self.cfg.embed_dim
        self.H = cfg.img_h // 8
        self.W = cfg.img_w // 8
        self.blocks = nn.ModuleList()
        for i in range(len(self.cfg.depth)):
             self.blocks.append(BasicLayer(dim = self.embed_dim* 2**i, num_heads = self.num_heads[i], depth = self.depth[i], downsample = True if (i < len(self.cfg.depth) - 1) else None, H = self.H // 2**i, W = self.W // 2**i, C = self.embed_dim* 2**i))

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)

        return x

