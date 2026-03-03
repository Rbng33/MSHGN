# ============================================================
# MSHGN V2 — Memory & Speed Optimized
# Target: <4GB memory, <1.5s/batch, similar performance
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os, gc, time
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_checkpoint



def log(msg="", flush=True):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=flush)


# ============================================================
# RevIN — unchanged (already efficient)
# ============================================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self.mean = self.stdev = None

    def forward(self, x, mode='norm', mask=None):
        if mode == 'norm':
            if mask is not None:
                s = mask.sum(-1, keepdim=True).clamp(min=1)
                self.mean = ((x * mask).sum(-1, keepdim=True) / s).detach()
                var = (((x - self.mean)**2 * mask).sum(-1, keepdim=True) / s)
            else:
                self.mean = x.mean(-1, keepdim=True).detach()
                var = x.var(-1, keepdim=True, unbiased=False)
            self.stdev = (var + self.eps).sqrt().detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.weight + self.bias
            return x
        else:
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            return x * self.stdev + self.mean


# ============================================================
# SeriesDecomp — unchanged
# ============================================================
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_p = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_p.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend


# ============================================================
# ObservationEmbedding — reduced to d=64
# ============================================================
class ObservationEmbedding(nn.Module):
    def __init__(self, num_channels, seq_len, d_model):
        super().__init__()
        self.d = d_model
        self.value_proj = nn.Linear(1, d_model)
        self.channel_embed = nn.Embedding(num_channels, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.register_buffer('pos_embed', self._sinusoidal(seq_len, d_model))
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _sinusoidal(length, d):
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        pe = torch.zeros(length, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x, mask):
        B, C, L = x.shape
        val = self.value_proj(x.unsqueeze(-1))
        ch = self.channel_embed(
            torch.arange(C, device=x.device)).reshape(1, C, 1, self.d)
        ps = self.pos_embed[:L].reshape(1, 1, L, self.d)
        mt = self.mask_token.reshape(1, 1, 1, self.d)
        m = mask.unsqueeze(-1)
        return self.norm(val * m + mt * (1 - m) + ch + ps)


# ============================================================
# FIX 1: Lightweight LocalConv — use ONLY depthwise (no pointwise)
# Memory saved: removes one (BC, d, L) tensor saved for backward
# ============================================================
class LocalTemporalConv(nn.Module):
    def __init__(self, d_model, kernel_size=9, dropout=0.1):
        super().__init__()
        # CHANGE: Depthwise only, no pointwise d² conv
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size,
                      padding=kernel_size // 2, groups=d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, L, d = x.shape
        out = self.conv(x.reshape(B * C, L, d).transpose(1, 2))
        return x + self.norm(out.transpose(1, 2).reshape(B, C, L, d))


# ============================================================
# FIX 2: Lightweight MultiscaleMixing
# Changes:
#   - scales: 3 → 2 (saves 25% compute + memory)
#   - Removed trend_up convs (trends are smooth, interpolation suffices)
#   - Shared single conv for season_down and season_up
#   - FFN expansion: 4× → 2× (halves biggest saved tensor)
#   - Grouped pointwise convs (groups=4)
#
# V1: 9 conv blocks + 4× FFN = ~2.8 GB saved tensors
# V2: 4 conv blocks + 2× FFN = ~0.6 GB saved tensors
# ============================================================
class EfficientMultiscaleMixing(nn.Module):
    def __init__(self, d_model, num_scales=2, kernel_size=9, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales
        self.decomp = SeriesDecomp(kernel_size)

        # CHANGE: Lightweight conv — grouped pointwise
        def _conv():
            return nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, 1, groups=4),  # grouped!
                nn.Dropout(dropout),
            )

        # CHANGE: Only season_down and season_up (no trend convs)
        self.season_down = nn.ModuleList(
            [_conv() for _ in range(num_scales)])
        self.season_up = nn.ModuleList(
            [_conv() for _ in range(num_scales)])
        # Trends: just interpolation, no learned convs

        # CHANGE: 2× expansion instead of 4×
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, L, d = x.shape
        h = x.reshape(B * C, L, d)

        # Build multi-scale pyramid
        scales = [h]
        cur = h.transpose(1, 2)
        for _ in range(self.num_scales):
            cur = F.avg_pool1d(cur, 2, 2)
            scales.append(cur.transpose(1, 2))

        # Decompose at each scale
        seasons, trends = [], []
        for s in scales:
            sea, tre = self.decomp(s)
            seasons.append(sea.transpose(1, 2))
            trends.append(tre.transpose(1, 2))

        # Season: down then up (with learned convs)
        for i in range(self.num_scales):
            dn = F.avg_pool1d(seasons[i], 2, 2)
            seasons[i + 1] = seasons[i + 1] + self.season_down[i](dn)

        for i in reversed(range(self.num_scales)):
            up = F.interpolate(seasons[i + 1], size=seasons[i].shape[-1],
                               mode='linear', align_corners=False)
            seasons[i] = seasons[i] + self.season_up[i](up)

        # CHANGE: Trend — just interpolate, no convs
        for i in reversed(range(self.num_scales)):
            up = F.interpolate(trends[i + 1], size=trends[i].shape[-1],
                               mode='linear', align_corners=False)
            trends[i] = trends[i] + up  # simple addition, no conv

        out = (seasons[0] + trends[0]).transpose(1, 2)
        out = h + self.ffn(self.norm(out))
        return out.reshape(B, C, L, d)


# ============================================================
# TemporalHyperedgeUpdate — slightly streamlined
# ============================================================
class TemporalHyperedgeUpdate(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.H = n_heads
        self.dh = d_model // n_heads
        self.scale = self.dh ** -0.5
        # CHANGE: Fused QKV projection (one matmul instead of three)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, C, L, d = x.shape
        BL = B * L
        xt = x.permute(0, 2, 1, 3).reshape(BL, C, d)

        # Fused QKV
        qkv = self.qkv(xt).reshape(BL, C, 3, self.H, self.dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BL, H, C, dh)
        q, k, v = qkv.unbind(0)

        sc = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            mk = mask.permute(0, 2, 1).reshape(BL, C)
            sc = sc.masked_fill(
                mk[:, None, None, :] == 0, torch.finfo(sc.dtype).min)
        att = self.drop(torch.softmax(sc, -1))
        out = self.o_proj(
            torch.matmul(att, v).transpose(1, 2).reshape(BL, C, d))

        if mask is not None:
            w = mask.permute(0, 2, 1).reshape(BL, C, 1)
            E = (out * w).sum(1) / w.sum(1).clamp(min=1)
        else:
            E = out.mean(1)
        return self.norm(E.reshape(B, L, d))


# ============================================================
# ChannelHyperedgeInteraction — 2× FFN instead of 4×
# ============================================================
class ChannelHyperedgeInteraction(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d = d_model
        # Fused QKV
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # CHANGE: 2× expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout))
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, C, L, d = x.shape
        if mask is not None:
            w = mask.unsqueeze(-1)
            E = (x * w).sum(2) / w.sum(2).clamp(min=1)
        else:
            E = x.mean(2)

        qkv = self.qkv(E).reshape(B, C, 3, d).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)

        S = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
        if mask is not None:
            joint = torch.matmul(mask, mask.transpose(1, 2))
            total = mask.sum(-1)
            tp = total.unsqueeze(-1) + total.unsqueeze(-2)
            overlap = 2 * joint / tp.clamp(min=1)
            S = S * (0.5 + 0.5 * overlap)

        out = self.o_proj(
            torch.matmul(self.drop(torch.softmax(S, -1)), v))
        E = self.n1(E + out)
        return self.n2(E + self.ffn(E))


# ============================================================
# HyperedgeToNode — 2× FFN
# ============================================================
class HyperedgeToNode(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.t_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        # CHANGE: 2× expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, E_t, E_c):
        out = x + self.t_proj(E_t).unsqueeze(1) + self.c_proj(E_c).unsqueeze(2)
        return out + self.ffn(self.norm(out))


# ============================================================
# MSHGNBlock — same structure, uses optimized components
# ============================================================
class MSHGNBlock(nn.Module):
    def __init__(self, d_model, num_scales=2, conv_kernel=9,
                 n_heads=4, dropout=0.1, use_checkpoint=True,
                 use_amp=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp
        self.local_conv = LocalTemporalConv(d_model, conv_kernel, dropout)
        self.multiscale = EfficientMultiscaleMixing(
            d_model, num_scales, conv_kernel, dropout)
        self.temporal_he = TemporalHyperedgeUpdate(
            d_model, n_heads, dropout)
        self.channel_he = ChannelHyperedgeInteraction(d_model, dropout)
        self.he_to_node = HyperedgeToNode(d_model, dropout)

    def _inner_forward(self, x, mask):
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            x = self.local_conv(x)
            x = self.multiscale(x)
            E_t = self.temporal_he(x, mask)
            E_c = self.channel_he(x, mask)
            return self.he_to_node(x, E_t, E_c)

    def forward(self, x, mask=None):
        if self.use_checkpoint and self.training:
            return grad_checkpoint(
                self._inner_forward, x, mask, use_reentrant=False)
        return self._inner_forward(x, mask)


# ============================================================
# MSHGN V2 — main model
# ============================================================
class MSHGN(nn.Module):
    def __init__(self, num_channels, seq_len, d_model=64,
                 num_layers=3, num_scales=2, conv_kernel=9,
                 n_heads=4, dropout=0.15, use_checkpoint=True,
                 use_amp=True):
        super().__init__()
        self.C = num_channels
        self.L = seq_len
        self.use_amp = use_amp
        self.revin = RevIN(num_channels, affine=True)
        self.embed = ObservationEmbedding(num_channels, seq_len, d_model)
        self.blocks = nn.ModuleList([
            MSHGNBlock(d_model, num_scales, conv_kernel, n_heads, dropout,
                       use_checkpoint=use_checkpoint, use_amp=use_amp)
            for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        if x.dim() == 3 and x.shape[1] == self.L and x.shape[2] == self.C:
            x = x.transpose(1, 2)
        if mask is None:
            mask = torch.ones_like(x)
        x = self.revin(x, 'norm', mask)
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            h = self.embed(x, mask)
        for blk in self.blocks:
            h = blk(h, mask)
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            out = self.head(h).squeeze(-1)
        return self.revin(out.float(), 'denorm')

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
