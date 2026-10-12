"""Build the PyTorch models and load them straight from the official Flax checkpoints."""

import os
import re

import torch

from . import ops as default_ops
from .config import DalleBartConfig, VQGANConfig
from .flax_io import FlaxCheckpoint
from .model import DalleBart, RMSNorm
from .vqgan import VQGANDecoder

WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"

# Flax leaf name -> torch parameter name
_LEAVES = {"kernel": "weight", "embedding": "weight", "scale": "weight", "bias": "bias", "head_scale": "head_scale"}
_LAYER_KEY = re.compile(r"FlaxBart(?:Encoder|Decoder)Layer(?:s|_(\d+))$")


def _to_torch(leaf, array, dtype):
    """Flax array -> torch tensor in torch layout. Always copies, so nothing references the checkpoint file."""
    tensor = torch.from_numpy(array)
    if leaf == "kernel":
        if tensor.ndim == 2:  # dense: [in, out] -> [out, in]
            tensor = tensor.T
        elif tensor.ndim == 4:  # conv: [H, W, in, out] -> [out, in, H, W]
            tensor = tensor.permute(3, 2, 0, 1)
    elif leaf == "head_scale":  # [1, 1, heads, 1] -> [heads, 1, 1]
        tensor = tensor.reshape(-1, 1, 1)
    return tensor.to(dtype=dtype, copy=True).contiguous()


def _resolve(module, flax_path):
    """Follow a Flax module path through the torch tree, translating Flax's auto-assigned names."""
    names = []
    for flax_name in flax_path:
        name = getattr(module, "flax_names", {}).get(flax_name, flax_name)
        child = getattr(module, name, None)
        if not isinstance(child, torch.nn.Module):
            raise ValueError(f"checkpoint module {'/'.join(flax_path)} has no counterpart in the model; "
                             "the weights do not match config.json")
        module = child
        names.append(name)
    return ".".join(names)


def _param_key(module_name, leaf, path):
    if leaf not in _LEAVES:
        raise ValueError(f"unexpected checkpoint entry {'/'.join(path)}")
    return f"{module_name}.{_LEAVES[leaf]}"


def _dalle_bart_state_dict(model, flat, dtype):
    state = {}
    for path, array in flat.items():
        *modules, leaf = path
        if modules[0] == "model":
            modules = modules[1:]
        if len(modules) > 2 and modules[1] == "layers":
            stack_name, _, layer_key, *inner = modules
            match = _LAYER_KEY.match(layer_key)
            if match is None:
                raise ValueError(f"unexpected layer collection entry {'/'.join(path)}")
            layers = getattr(model, stack_name).layers
            if match.group(1) is None:
                # nn.scan checkpoint: one array holds every layer, stacked on a leading axis
                if array.shape[0] != len(layers):
                    raise ValueError(f"{'/'.join(path)} holds {array.shape[0]} layers, config.json says {len(layers)}")
                per_layer = list(enumerate(array))
            else:
                per_layer = [(int(match.group(1)), array)]
            for index, layer_array in per_layer:
                if index >= len(layers):
                    raise ValueError(f"{'/'.join(path)}: config.json says there are only {len(layers)} layers")
                name = f"{stack_name}.layers.{index}.{_resolve(layers[index], inner)}"
                state[_param_key(name, leaf, path)] = _to_torch(leaf, layer_array, dtype)
        else:
            state[_param_key(_resolve(model, modules), leaf, path)] = _to_torch(leaf, array, dtype)
    return state


def _vqgan_decoder_state_dict(model, flat, dtype):
    state = {}
    for path, array in flat.items():
        *modules, leaf = path
        if modules[0] in ("encoder", "quant_conv"):
            continue  # only needed to turn images into tokens
        state[_param_key(_resolve(model, modules), leaf, path)] = _to_torch(leaf, array, dtype)
    return state


def _add_implicit_norm_scales(model, state, dtype):
    """Flax leaves out the scale of a norm that feeds a dense layer; that is a scale of one."""
    for name, module in model.named_modules():
        if f"{name}.weight" in state:
            continue
        if isinstance(module, torch.nn.LayerNorm):
            state[f"{name}.weight"] = torch.ones(module.normalized_shape, dtype=dtype)
        elif isinstance(module, RMSNorm):
            state[f"{name}.weight"] = torch.ones(module.weight.shape, dtype=dtype)


def _load(model_class, config_class, to_state_dict, directory, dtype, device, operations):
    config_path = os.path.join(directory, CONFIG_NAME)
    weights_path = os.path.join(directory, WEIGHTS_NAME)
    for path in (config_path, weights_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found")
    config = config_class.from_json(config_path)
    model = model_class(config, dtype=dtype, device=device, operations=operations)
    with FlaxCheckpoint(weights_path) as checkpoint:
        flat = checkpoint.flat()
        state = to_state_dict(model, flat, dtype)
        del flat
    _add_implicit_norm_scales(model, state, dtype)
    # assign: the freshly converted tensors become the parameters, so the weights exist in memory once
    model.load_state_dict(state, strict=True, assign=True)
    if device is not None:
        model.to(device)
    return model.eval().requires_grad_(False)


def load_dalle_bart(directory, dtype=torch.float32, device=None, operations=default_ops):
    """Load DalleBart from a folder holding ``config.json`` and ``flax_model.msgpack``."""
    return _load(DalleBart, DalleBartConfig, _dalle_bart_state_dict, directory, dtype, device, operations)


def load_vqgan_decoder(directory, dtype=torch.float32, device=None, operations=default_ops):
    """Load the VQGAN decoder from a folder holding ``config.json`` and ``flax_model.msgpack``."""
    return _load(VQGANDecoder, VQGANConfig, _vqgan_decoder_state_dict, directory, dtype, device, operations)
