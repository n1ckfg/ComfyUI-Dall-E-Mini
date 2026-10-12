"""PyTorch port of the decoding half of vqgan-jax's ``VQModel`` (taming-transformers VQGAN).

Only what turns image tokens into pixels is built: codebook, post-quant conv and decoder.
Attribute names match the Flax parameter tree (``up_4``, ``block_0``, ``attn_1`` ...), so
checkpoints convert by path alone.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops as default_ops
from .config import VQGANConfig


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, operations, dd):
        super().__init__()
        self.norm1 = operations.GroupNorm(32, in_channels, eps=1e-6, **dd)
        self.conv1 = operations.Conv2d(in_channels, out_channels, 3, padding=1, **dd)
        self.norm2 = operations.GroupNorm(32, out_channels, eps=1e-6, **dd)
        self.conv2 = operations.Conv2d(out_channels, out_channels, 3, padding=1, **dd)
        self.nin_shortcut = operations.Conv2d(in_channels, out_channels, 1, **dd) if in_channels != out_channels else None

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + (x if self.nin_shortcut is None else self.nin_shortcut(x))


class AttnBlock(nn.Module):
    def __init__(self, channels, operations, dd):
        super().__init__()
        self.norm = operations.GroupNorm(32, channels, eps=1e-6, **dd)
        self.q = operations.Conv2d(channels, channels, 1, **dd)
        self.k = operations.Conv2d(channels, channels, 1, **dd)
        self.v = operations.Conv2d(channels, channels, 1, **dd)
        self.proj_out = operations.Conv2d(channels, channels, 1, **dd)

    def forward(self, x):
        b, c, height, width = x.shape
        h = self.norm(x)
        # single-head attention over the height * width positions
        q, k, v = (conv(h).view(b, 1, c, height * width).transpose(2, 3) for conv in (self.q, self.k, self.v))
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(2, 3).reshape(b, c, height, width)
        return x + self.proj_out(h)


class Upsample(nn.Module):
    def __init__(self, channels, with_conv, operations, dd):
        super().__init__()
        self.conv = operations.Conv2d(channels, channels, 3, padding=1, **dd) if with_conv else None

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return x if self.conv is None else self.conv(x)


class MidBlock(nn.Module):
    def __init__(self, channels, operations, dd):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels, operations, dd)
        self.attn_1 = AttnBlock(channels, operations, dd)
        self.block_2 = ResnetBlock(channels, channels, operations, dd)

    def forward(self, x):
        return self.block_2(self.attn_1(self.block_1(x)))


class UpsamplingBlock(nn.Module):
    def __init__(self, cfg, level, resolution, operations, dd):
        super().__init__()
        last = cfg.num_resolutions - 1
        block_in = cfg.ch * cfg.ch_mult[last if level == last else level + 1]
        block_out = cfg.ch * cfg.ch_mult[level]
        self.num_blocks = cfg.num_res_blocks + 1
        self.has_attn = resolution in cfg.attn_resolutions
        for i in range(self.num_blocks):
            setattr(self, f"block_{i}", ResnetBlock(block_in, block_out, operations, dd))
            block_in = block_out
            if self.has_attn:
                setattr(self, f"attn_{i}", AttnBlock(block_in, operations, dd))
        self.upsample = Upsample(block_in, cfg.resamp_with_conv, operations, dd) if level != 0 else None

    def forward(self, x):
        for i in range(self.num_blocks):
            x = getattr(self, f"block_{i}")(x)
            if self.has_attn:
                x = getattr(self, f"attn_{i}")(x)
        return x if self.upsample is None else self.upsample(x)


class Decoder(nn.Module):
    def __init__(self, cfg, operations, dd):
        super().__init__()
        self.num_levels = cfg.num_resolutions
        last = self.num_levels - 1
        block_in = cfg.ch * cfg.ch_mult[last]
        self.conv_in = operations.Conv2d(cfg.z_channels, block_in, 3, padding=1, **dd)
        self.mid = MidBlock(block_in, operations, dd)
        for level in range(self.num_levels):
            # level `last` works on the token grid; each lower level doubles the resolution
            setattr(self, f"up_{level}", UpsamplingBlock(cfg, level, cfg.resolution // 2**level, operations, dd))
        self.norm_out = operations.GroupNorm(32, cfg.ch * cfg.ch_mult[0], eps=1e-6, **dd)
        self.conv_out = operations.Conv2d(cfg.ch * cfg.ch_mult[0], cfg.out_ch, 3, padding=1, **dd)

    def forward(self, x):
        x = self.mid(self.conv_in(x))
        for level in reversed(range(self.num_levels)):
            x = getattr(self, f"up_{level}")(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class Quantizer(nn.Module):
    def __init__(self, cfg, operations, dd):
        super().__init__()
        self.embedding = operations.Embedding(cfg.n_embed, cfg.embed_dim, **dd)


class VQGANDecoder(nn.Module):
    def __init__(self, cfg: VQGANConfig, dtype=None, device=None, operations=default_ops):
        super().__init__()
        self.config = cfg
        dd = {"dtype": dtype, "device": device}
        self.quantize = Quantizer(cfg, operations, dd)
        self.post_quant_conv = operations.Conv2d(cfg.embed_dim, cfg.z_channels, 1, **dd)
        self.decoder = Decoder(cfg, operations, dd)

    @torch.no_grad()
    def decode_tokens(self, tokens):
        """[B, N] codebook indices (N a square number) -> [B, H, W, 3] images in [0, 1]."""
        batch, count = tokens.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError(f"expected a square number of image tokens, got {count}")
        if tokens.numel() and (int(tokens.min()) < 0 or int(tokens.max()) >= self.config.n_embed):
            raise ValueError(f"image tokens must be in [0, {self.config.n_embed})")
        z = self.quantize.embedding(tokens).view(batch, side, side, -1).permute(0, 3, 1, 2)
        images = self.decoder(self.post_quant_conv(z))
        # as upstream: this VQGAN's output is clipped to [0, 1] directly, with no [-1, 1] rescale
        return images.permute(0, 2, 3, 1).clamp(0.0, 1.0)
