# -*- coding: utf-8 -*-
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from basicsr.utils.registry import ARCH_REGISTRY

# PFCA change the name to LECA
def math_std(x):
    a = 1e-4
    x_mean = x.mean(1, keepdim=True)
    x_norm = (x - x_mean).pow(2)
    x_std = torch.std(x, dim=1, keepdim=True) + a
    x_out = (x_norm + 2 * x_std) / 4 * x_std
    return x_out

class PFCA(nn.Module):
    def __init__(self):
        super(PFCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.shared_MLP = nn.Sequential(
        #     nn.Conv2d(channel, channel // ratio, 1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(channel // ratio, channel, 1, bias=False)
        # )
        self.sigmoid = nn.Sigmoid()
        # self.math = math_std()
    def forward(self, x):
        avgout = math_std(self.avg_pool(x))
        # maxout = self.shared_MLP(self.max_pool(x))
        x = self.sigmoid(avgout) * x
        return x

# PFSA change the name to DLESA
def math_std_SA(x):
    a = 1e-4
    x_mean_w = x.mean(2, keepdim=True)
    x_norm = (x - x_mean_w).pow(2)
    x_std = torch.std(x, dim=2, keepdim=True) + a
    x_out_w = (x_norm + 2 * x_std) / 4 * x_std

    x_mean_h = x.mean(3, keepdim=True)
    x_norm = (x - x_mean_h).pow(2)
    x_std = torch.std(x, dim=3, keepdim=True) + a
    x_out_h = (x_norm + 2 * x_std) / 4 * x_std

    x_out = x_out_w + x_out_h
    return x_out

class PFSA(nn.Module):
    def __init__(self):
        super(PFSA, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = math_std_SA(x)
        x = self.sigmoid(avgout) * x
        return x
# HAB  change the name to KIPM
class HAB(nn.Module):
    def __init__(self, num_feat):
        super(HAB, self).__init__()
        self.norm = LayerNorm(num_feat, data_format='channels_first')
        self.conv_first = nn.Conv2d(num_feat, 2*num_feat, 1, 1, 0)
        self.PFCA = PFCA()
        self.PFSA = PFSA()
        self.X7 = nn.Conv2d(num_feat, num_feat, 7, 1, 7//2, groups=num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x.clone()
        x = self.norm(x)
        x = self.conv_first(x)
        a, b = torch.chunk(x, 2, dim=1)
        a_1 = self.PFCA(a)
        b_1 = self.PFSA(b)
        x = a_1 + b_1
        x = self.X7(x)
        x = self.conv_last(x)
        x = self.sigmoid(x) * res
        return x

# MLP change the name to FFN
class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        i_feats = 2 * n_feats

        self.fc1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.act = SimpleGate()
        self.fc2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x + shortcut

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Large Kernel Attention Branch (LKAB)
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=7 // 2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 9, stride=1, padding=((9 // 2) * 4), groups=dim, dilation=4)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

# Ablation Studies CA
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

        # self.attention = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

    def forward(self, x):
        y = self.attention(x)
        return x * y

# Ablation Studies Resblock
class PFRCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(PFRCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            PFCA(), PFSA())

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x

# Detail Feature Enhancement Module (DFEM)
class GroupGLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 3 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats
        self.norm = LayerNorm(n_feats, data_format='channels_first')

        # self.LKA9 = nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats,
        #                       dilation=4)
        self.LKA7 = nn.Conv2d(n_feats, n_feats, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats,
                              dilation=3)
        self.LKA5 = nn.Conv2d(n_feats, n_feats, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats,
                              dilation=2)
        # self.LKA7 = nn.Conv2d(n_feats, n_feats, 7, stride=1, padding=(7 // 2), groups=n_feats)
        # self.LKA5 = nn.Conv2d(n_feats, n_feats, 5, stride=1, padding=(5 // 2), groups=n_feats)

        self.conv2 = nn.Conv2d(self.n_feats, self.n_feats, 5, 1, (5 // 2), groups=self.n_feats)
        # self.conv2 = nn.Conv2d(self.n_feats, self.n_feats, 1, 1, 0)

        self.conv3 = nn.Conv2d(self.n_feats, self.n_feats, 9, 1, (9 // 2), groups=n_feats)

        self.conv4 = nn.Conv2d(self.n_feats, self.n_feats, 1, 1, 0)

        self.conv5 = nn.Conv2d(self.n_feats, self.n_feats, 1, 1, 0)

        self.conv6 = nn.Conv2d(self.n_feats * 2, self.n_feats, 1, 1, 0)

        self.atten = LKA(self.n_feats)

        self.act1 = nn.Hardswish()
        self.act2 = nn.Tanh()

        # # Ablation Studies Sigmoid()
        # self.act2 = nn.Sigmoid()

    def forward(self, x):
        shortcut = x.clone()
        x0 = self.norm(x)

        x3 = self.atten(x0)

        x5 = self.conv2(x0)
        x6 = self.conv3(x0)
        x7 = self.LKA5(x5).mul(self.LKA7(x5))
        x8 = self.act2(self.conv5(self.act1(self.conv4(x7))))

        x9 = x8.mul(x6)

        y = self.conv6(torch.cat((x3, x9), dim=1)) + shortcut

        return y

# Progressive  Feature Aggregation Block（PFAB）
class MAB(nn.Module):
    def __init__(
            self, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.HAB = HAB(self.n_feats)
        self.LFE = MLP(self.n_feats)
        # self.conv = nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1)
        self.LKA = GroupGLKA(self.n_feats)
        self.LFE1 = MLP(self.n_feats)

    def forward(self, x):
        # large kernel attention
        res1 = x.clone()
        x = self.HAB(x)
        # local feature extraction
        x = self.LFE(x)
        x = self.LKA(x)
        x = self.LFE1(x)
        return x + res1


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


# PFAN: Progressive  Feature Aggregation Network
#    n_feats : 64 for PFAN  32 for PFAN-tiny
# n_resgroups: 10 for PFAN   6 for PFAN-tiny
# we change the name 'GLAN' to 'PFAN'

@ARCH_REGISTRY.register()
class GLAN(nn.Module):
    def __init__(self, n_resgroups=10, n_colors=3, n_feats=64, scale=4):
        super(GLAN, self).__init__()

        # res_scale = res_scale
        self.n_resgroups = n_resgroups
        self.scale = scale
        self.sub_mean = MeanShift(1.0)
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        # define body module
        self.body = nn.ModuleList([
            MAB(n_feats)
            for i in range(n_resgroups)])

        self.body_t = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )
        self.add_mean = MeanShift(1.0, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        # y = F.interpolate(x, (x.size(-2) * self.scale, x.size(-1) * self.scale), mode='bicubic', align_corners=False)
        y = F.interpolate(x, size=[x.shape[2] * self.scale, x.shape[3] * self.scale], mode="bilinear", align_corners=False)
        # mode="bilinear"
        x = self.head(x)
        res = x
        for i in self.body:
            res = i(res)
        res = self.body_t(res) + x
        x = self.tail(res)
        x = x + y
        x = self.add_mean(x)
        return x


# if __name__ == '__main__':
#     # torch.cuda.empty_cache()
#     net = GLAN.cuda()
#     x = torch.randn((1, 3, 16, 16)).cuda()
#     x = net(x)
#     print(x.shape)
