'''
Import packages

'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

#define the modules and models:

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Sequential(
            Rearrange('b c u a -> b u a c'),
            nn.Linear(dim, inner_dim * 3, bias = False)
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b u a (h d) -> b h u a d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h u a d -> b u a (h d)')
        return rearrange(self.to_out(out), 'b u a c -> b c u a')

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange('b c u a -> b u a c'),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
            Rearrange('b u a c -> b c u a')
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, hidden_dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.atn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, hidden_dim, dropout)
    def forward(self, x):
        x = self.atn(x) + x
        x = self.ff(x) + x
        return x


class DepthwiseConv(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )
    def forward(self, x):
        x = self.layer(x) + x
        return x

class PointwiseConv(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(indim, outdim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(outdim)
        )
        self.indim = indim
        self.outdim = outdim
    def forward(self, x):
        if self.indim == self.outdim:
            x = self.layer(x) + x
        else:
            x = self.layer(x)
        return x


class Encodings(nn.Module):
    def __init__(self, *,  image_size, patch_size, dim, channels = 3, dim_head = 64, emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        
        patches_up, patches_across = (image_height // patch_height), (image_width // patch_width)

        patch_dim = channels * patch_height * patch_width
    
        

        assert dim >= patch_dim

        

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
            Rearrange('b (h w) (d) -> b h w (d)', h = patches_up, w = patches_across)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, patches_up, patches_across, dim))

        self.dropout = nn.Dropout(emb_dropout)
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, h, w, _ = x.shape
        x += self.pos_embedding
        x = self.dropout(x)
        return rearrange(x, 'b h w c -> b c h w')

class CT_block_inline(nn.Module):
    def __init__(self, *, dim, hidden_dim, kernel_size, indim, outdim, heads = 8, dim_head = 64, dropout = 0):
        super().__init__()
        
        self.T = Transformer(dim = dim, hidden_dim = dim)
        self.DC = DepthwiseConv(dim = dim, kernel_size = kernel_size)
        self.PC = PointwiseConv(indim = indim, outdim = outdim)
    def forward(self, x):
        
        x = self.DC(x)
        x = self.T(x)
        x = self.PC(x)
        return x

class CT_block_parallel_concat(nn.Module):
    def __init__(self, *, dim, hidden_dim, kernel_size, outdim, indim = 0, heads = 8, dim_head = 64, dropout = 0):
        super().__init__()
        self.T = Transformer(dim = dim, hidden_dim = dim)
        self.DC = DepthwiseConv(dim = dim, kernel_size = kernel_size)
        self.PC = PointwiseConv(indim = 2*outdim, outdim = outdim)
    def forward(self, x):
        x1 = self.DC(x)
        x2 = self.T(x)
        x = torch.concat((x1, x2), dim = 1)
        x = self.PC(x)
        return x

class CT_block_parallel_mm(nn.Module):
    def __init__(self, *, dim, hidden_dim, kernel_size, outdim, indim = 0, heads = 8, dim_head = 64, dropout = 0):
        super().__init__()
        self.T = Transformer(dim = dim, hidden_dim = dim)
        self.DC = DepthwiseConv(dim = dim, kernel_size = kernel_size)
        self.PC = PointwiseConv(indim = indim, outdim = outdim)
    def forward(self, x):
        x1 = self.DC(x)
        x2 = self.T(x)
        x = x1@x2
        x = self.PC(x)
        return x

class PCT_block(nn.Module):
    def __init__(self, *, dim, hidden_dim, kernel_size, outdim, indim = 0, heads = 8, dim_head = 64, dropout = 0):
        super().__init__()
        self.T = Transformer(dim = dim, hidden_dim = dim)
        self.PC = PointwiseConv(indim = indim, outdim = outdim)
    def forward(self, x):
        x = self.T(x)
        x = self.PC(x)
        return x

class ConvMixer_block(nn.Module):
    def __init__(self, *, dim, hidden_dim, kernel_size, outdim, indim = 0, heads = 8, dim_head = 64, dropout = 0):
        super().__init__()
        self.DC = DepthwiseConv(dim = dim, kernel_size = kernel_size)
        self.PC = PointwiseConv(indim = indim, outdim = outdim)
    def forward(self, x):
        x = self.DC(x)
        x = self.PC(x)
        return x


class Model(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, hidden_dim, kernel_size, indim, outdim, numblocks, block_type, heads = 8, dim_head = 64, dropout = 0,channels = 3,  emb_dropout = 0., num_classes = 10):
        super().__init__()
        block_types = {'inline':CT_block_inline, 'concat':CT_block_parallel_concat, 'mm':CT_block_parallel_mm, 'pct': PCT_block, 'cm':ConvMixer_block}
        CT_block = block_types[block_type]
        self.enc = Encodings(image_size = image_size, patch_size = patch_size, dim = dim, channels = channels)
        self.CTB = nn.ModuleList([CT_block(dim = dim, hidden_dim = hidden_dim, kernel_size = kernel_size, indim = indim, outdim = outdim) for i in range(numblocks)])
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, 100),
            nn.GELU(),
            nn.Linear(100, num_classes)
        )
    def forward(self, x):
        x = self.enc(x)
        for block in self.CTB:
            x = block(x) + x
        x = self.final(x)
        return x

