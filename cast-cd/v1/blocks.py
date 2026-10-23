"""Fusion, attention, and decoder building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat
from torch.nn import (Conv2d, InstanceNorm2d, Module, PReLU, Sequential,
                      Sigmoid, Upsample)


class PixelwiseLinear(Module):
    def __init__(self, fin, fout, last_activation=None):
        assert len(fout) == len(fin)
        super().__init__()
        n = len(fin)
        self._linears = Sequential(
            *[Sequential(Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                         PReLU() if i < n-1 or last_activation is None else last_activation)
              for i in range(n)])
    def forward(self, x):
        return self._linears(x)


class SqueezeExcitation(Module):
    """Channel attention mechanism."""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.se = Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x)


class GDAF(Module):
    """
    Gated Difference-Attention Fusion (GDAF)
    Replaces the simple MixingBlock. Computes absolute difference,
    applies channel attention, and blends with concatenated features.
    """
    def __init__(self, channels):
        super().__init__()
        self.se = SqueezeExcitation(channels, reduction=4)
        # Gate: learns how much difference vs. concat to use
        # Input to gate is [f1, f2, att_diff], so 3*channels
        self.gate = Sequential(Conv2d(channels*3, channels, 1), Sigmoid())
        # Final projection with PReLU and InstanceNorm
        self.proj = Sequential(
            Conv2d(channels, channels, 3, padding=1, groups=channels),
            InstanceNorm2d(channels),
            PReLU(),
            Conv2d(channels, channels, 1)
        )

    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        att_diff = self.se(diff)
        gate = self.gate(cat([f1, f2, att_diff], dim=1))
        fused = gate * att_diff + (1 - gate) * ((f1 + f2) / 2.0)
        return self.proj(fused)


class CrossTemporalAttention(Module):
    """
    Lightweight Cross-Temporal Channel Attention.
    Computes cross-attention across the channel dimension instead of spatial dimension.
    This reduces complexity from O(H^2 W^2) to O(C^2), allowing it to process
    1024x1024 high-resolution images without Out-Of-Memory errors!
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = Conv2d(channels, channels, 1, bias=False)
        self.k_proj = Conv2d(channels, channels, 1, bias=False)
        self.v_proj = Conv2d(channels, channels, 1, bias=False)

        self.out_proj = Sequential(
            Conv2d(channels*2, channels, 1),
            InstanceNorm2d(channels),
            PReLU()
        )

    def _cross_attn(self, q_feat, k_feat):
        B, C, H, W = q_feat.shape

        Q = self.q_proj(q_feat).view(B, self.num_heads, C // self.num_heads, H * W)
        K = self.k_proj(k_feat).view(B, self.num_heads, C // self.num_heads, H * W)
        V = self.v_proj(k_feat).view(B, self.num_heads, C // self.num_heads, H * W)

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        attn = (Q @ K.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).view(B, C, H, W)
        return out

    def forward(self, f1, f2):
        attn_1to2 = self._cross_attn(f1, f2)
        attn_2to1 = self._cross_attn(f2, f1)
        return self.out_proj(cat([attn_1to2, attn_2to1], dim=1))


class UpConcat(Module):
    def __init__(self, scale_factor, in_channels_x, in_channels_y, nout):
        super().__init__()
        self._upsample = Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        nin = in_channels_x + in_channels_y if in_channels_y > 0 else in_channels_x
        self._convolution = Sequential(
            Conv2d(nin, nin, 3, 1, groups=nin, padding=1), PReLU(), InstanceNorm2d(nin),
            Conv2d(nin, nout, kernel_size=1, stride=1),    PReLU(), InstanceNorm2d(nout))

    def forward(self, x, y=None):
        x = self._upsample(x)
        if y is not None:
            if x.shape[2] != y.shape[2] or x.shape[3] != y.shape[3]:
                x = x[:, :, :y.shape[2], :y.shape[3]]
            x = torch.cat([x, y], dim=1)
        return self._convolution(x)
