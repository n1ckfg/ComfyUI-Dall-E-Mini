"""Fallback layer set used when the models are built outside ComfyUI.

Inside ComfyUI the nodes pass ``comfy.ops.manual_cast`` instead, which has the same layer
names and lets ComfyUI keep weights on one device/dtype while computing on another.
Weights always come from a checkpoint, so random initialisation is skipped.
"""

import torch.nn as nn


class Linear(nn.Linear):
    def reset_parameters(self):
        return None


class Embedding(nn.Embedding):
    def reset_parameters(self):
        return None


class LayerNorm(nn.LayerNorm):
    def reset_parameters(self):
        return None


class GroupNorm(nn.GroupNorm):
    def reset_parameters(self):
        return None


class Conv2d(nn.Conv2d):
    def reset_parameters(self):
        return None
