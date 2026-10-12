"""PyTorch port of DalleBart (``dalle_mini/model/modeling.py``), inference only.

The module tree follows the Flax one closely enough that checkpoints convert mechanically:
every block records the name Flax auto-assigned to each of its children in ``flax_names``
(see ``loading.py``).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops as default_ops
from .config import DalleBartConfig


def _smelu(x, beta=1.0):
    x = torch.where(x <= -beta, torch.zeros_like(x), x)
    return torch.where(x >= beta, x, torch.square(x + beta) / (4 * beta))


# transformers.modeling_flax_utils.ACT2FN, plus dalle-mini's smelu
ACTIVATIONS = {
    "gelu": F.gelu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
    "smelu": _smelu,
}


def _residual_gain(cfg, is_encoder):
    if not cfg.use_deepnet_scaling:
        return 1.0
    if is_encoder:
        return 0.81 * (cfg.encoder_layers**4 * cfg.decoder_layers) ** 0.0625
    return (3 * cfg.decoder_layers) ** 0.25


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * weight


class _FlaxBlock(nn.Module):
    """A block whose children were auto-named ``ClassName_N`` by Flax, in creation order."""

    def __init__(self):
        super().__init__()
        self.flax_names = {}
        self._flax_counts = {}

    def _add(self, attr, flax_class, module):
        index = self._flax_counts.get(flax_class, 0)
        self._flax_counts[flax_class] = index + 1
        self.flax_names[f"{flax_class}_{index}"] = attr
        setattr(self, attr, module)

    def _add_norm(self, attr, enabled, cfg, dim, factory):
        """Create ``attr`` as a norm layer if ``enabled``, else leave it as None (no Flax name is consumed)."""
        if enabled:
            self._add(attr, "RMSNorm" if cfg.ln_type == "rmsnorm" else "LayerNorm", _norm(cfg, dim, **factory))
        else:
            setattr(self, attr, None)


def _norm(cfg, dim, dtype=None, device=None, operations=default_ops):
    # Flax omits the scale of norms that feed a dense layer; the loader fills those with ones,
    # so every norm here carries its full set of parameters.
    if cfg.ln_type == "rmsnorm":
        return RMSNorm(dim, eps=1e-5, dtype=dtype, device=device)
    return operations.LayerNorm(dim, eps=1e-5, dtype=dtype, device=device)


def _maybe(module, x):
    return x if module is None else module(x)


class Attention(nn.Module):
    def __init__(self, cfg, heads, q_length, k_length, is_encoder, is_cross, factory):
        super().__init__()
        dim = cfg.d_model
        if dim % heads != 0:
            raise ValueError(f"d_model ({dim}) must be divisible by the number of heads ({heads})")
        self.heads = heads
        self.head_dim = dim // heads
        self.k_length = k_length
        operations = factory["operations"]
        dd = {"dtype": factory["dtype"], "device": factory["device"]}

        self.q_proj = operations.Linear(dim, dim, bias=cfg.use_bias, **dd)
        self.k_proj = operations.Linear(dim, dim, bias=cfg.use_bias, **dd)
        self.v_proj = operations.Linear(dim, dim, bias=cfg.use_bias, **dd)
        self.out_proj = operations.Linear(dim, dim, bias=cfg.use_bias, **dd)

        self.head_scale = nn.Parameter(torch.empty(heads, 1, 1, **dd)) if cfg.use_head_scale else None
        self.tau = cfg.tau_init if cfg.use_cosine_attention else None
        self.rel_bias = operations.Embedding(q_length, k_length * heads, **dd) if cfg.use_swin_position_embeddings else None
        # Sinkhorn normalisation leaks future tokens through a causal mask, so it is encoder-only
        self.sinkhorn_iters = cfg.sinkhorn_iters if is_encoder else 1
        self.mid_layernorm = _norm(cfg, dim, **factory) if cfg.ln_positions == "subln" and not is_cross else None

    def _split_heads(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.head_dim).transpose(1, 2)

    def keys_values(self, x):
        return self._split_heads(self.k_proj(x)), self._split_heads(self.v_proj(x))

    def forward(self, x, keys, values, mask=None, q_start=0):
        """``x``: [B, Tq, D] queries. ``keys``/``values``: [B, H, Tk, D/H].
        ``mask``: bool, broadcastable to [B, H, Tq, Tk], True where attention is allowed.
        ``q_start``: position of the first query, only used by relative position biases."""
        b, tq, dim = x.shape
        q = self._split_heads(self.q_proj(x))

        if self.tau is None and self.rel_bias is None and self.sinkhorn_iters <= 1:
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        else:
            if self.tau is not None:
                q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
                keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
                weights = (q @ keys.transpose(-1, -2)) / self.tau
            else:
                weights = (q @ keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if mask is not None:
                weights = weights.masked_fill(~mask, float("-inf"))
            if self.rel_bias is not None:
                tk = keys.shape[-2]
                positions = torch.arange(q_start, q_start + tq, device=x.device)
                bias = self.rel_bias(positions).view(tq, self.k_length, self.heads)
                weights = weights + bias.permute(2, 0, 1)[:, :, :tk].to(weights.dtype)
            if self.sinkhorn_iters <= 1:
                weights = weights.softmax(dim=-1)
            else:
                for i in range(self.sinkhorn_iters):
                    weights = weights - weights.logsumexp(dim=-1 if i % 2 == 0 else -2, keepdim=True)
                    if mask is not None:
                        weights = weights.masked_fill(~mask, float("-inf"))
                weights = weights.exp()
            out = weights @ values

        if self.head_scale is not None:
            out = out * self.head_scale.to(device=out.device, dtype=out.dtype)
        out = out.transpose(1, 2).reshape(b, tq, dim)
        out = _maybe(self.mid_layernorm, out)
        return self.out_proj(out)


class FeedForward(_FlaxBlock):
    """Flax ``GLU`` (gated: act(W0 x) * (W1 x) -> W2) or ``FFN`` (act(W0 x) -> W1)."""

    def __init__(self, cfg, ffn_dim, factory):
        super().__init__()
        dim, pos = cfg.d_model, cfg.ln_positions
        operations = factory["operations"]
        dd = {"dtype": factory["dtype"], "device": factory["device"]}
        self.activation = ACTIVATIONS[cfg.activation_function]

        self._add_norm("norm_in", pos in ("normformer", "cogview", "preln", "subln"), cfg, dim, factory)
        self._add("fc_in", "Dense", operations.Linear(dim, ffn_dim, bias=cfg.use_bias, **dd))
        if cfg.use_glu:
            self._add("fc_gate", "Dense", operations.Linear(dim, ffn_dim, bias=cfg.use_bias, **dd))
        else:
            self.fc_gate = None
        self._add_norm("norm_mid", pos in ("normformer", "subln"), cfg, ffn_dim, factory)
        self._add("fc_out", "Dense", operations.Linear(ffn_dim, dim, bias=cfg.use_bias, **dd))
        self._add_norm("norm_out", pos in ("swinv2", "cogview"), cfg, dim, factory)

    def forward(self, x):
        x = _maybe(self.norm_in, x)
        h = self.activation(self.fc_in(x))
        if self.fc_gate is not None:
            h = h * self.fc_gate(x)
        h = _maybe(self.norm_mid, h)
        return _maybe(self.norm_out, self.fc_out(h))


class EncoderLayer(_FlaxBlock):
    def __init__(self, cfg, add_norm, factory):
        super().__init__()
        dim, pos = cfg.d_model, cfg.ln_positions
        self.res_gain = _residual_gain(cfg, is_encoder=True)

        self._add_norm("attn_pre_norm", pos in ("normformer", "cogview", "preln", "subln"), cfg, dim, factory)
        self._add("attn", "FlaxBartAttention", Attention(
            cfg, cfg.encoder_attention_heads, cfg.max_text_length, cfg.max_text_length,
            is_encoder=True, is_cross=False, factory=factory))
        self._add_norm("attn_post_norm", pos in ("normformer", "swinv2", "cogview"), cfg, dim, factory)
        self._add_norm("attn_res_norm", pos == "postln", cfg, dim, factory)
        self._add("ff", "GLU" if cfg.use_glu else "FFN", FeedForward(cfg, cfg.encoder_ffn_dim, factory))
        self._add_norm("final_norm", add_norm, cfg, dim, factory)

    def forward(self, x, mask):
        h = _maybe(self.attn_pre_norm, x)
        h = self.attn(h, *self.attn.keys_values(h), mask=mask)
        h = _maybe(self.attn_post_norm, h)
        x = _maybe(self.attn_res_norm, x * self.res_gain + h)
        x = x * self.res_gain + self.ff(x)
        return _maybe(self.final_norm, x)


class DecoderLayer(_FlaxBlock):
    def __init__(self, cfg, add_norm, factory):
        super().__init__()
        dim, pos = cfg.d_model, cfg.ln_positions
        heads = cfg.decoder_attention_heads
        self.res_gain = _residual_gain(cfg, is_encoder=False)
        pre = pos in ("normformer", "cogview", "preln")
        post = pos in ("normformer", "swinv2", "cogview")

        self._add_norm("self_pre_norm", pre, cfg, dim, factory)
        self._add("self_attn", "FlaxBartAttention", Attention(
            cfg, heads, cfg.image_length, cfg.image_length, is_encoder=False, is_cross=False, factory=factory))
        self._add_norm("self_post_norm", post, cfg, dim, factory)
        self._add_norm("self_res_norm", pos == "postln", cfg, dim, factory)

        self._add_norm("cross_pre_norm", pre, cfg, dim, factory)
        self._add("cross_attn", "FlaxBartAttention", Attention(
            cfg, heads, cfg.image_length, cfg.max_text_length, is_encoder=False, is_cross=True, factory=factory))
        self._add_norm("cross_post_norm", post, cfg, dim, factory)
        self._add_norm("cross_res_norm", pos == "postln", cfg, dim, factory)

        self._add("ff", "GLU" if cfg.use_glu else "FFN", FeedForward(cfg, cfg.decoder_ffn_dim, factory))
        self._add_norm("final_norm", add_norm, cfg, dim, factory)

    def forward(self, x, cross_kv, cross_mask, cache=None, position=0):
        """Without ``cache`` this is a full causal pass over ``x``. With ``cache`` (a pair of
        [B, H, image_length, D/H] buffers) ``x`` holds only the token at ``position`` and attends
        to the keys/values cached for all earlier positions."""
        t = x.shape[1]
        h = _maybe(self.self_pre_norm, x)
        keys, values = self.self_attn.keys_values(h)
        if cache is None:
            mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril_()
        else:
            end = position + t
            cache[0][:, :, position:end] = keys
            cache[1][:, :, position:end] = values
            keys, values = cache[0][:, :, :end], cache[1][:, :, :end]
            mask = None if t == 1 else torch.ones(t, end, dtype=torch.bool, device=x.device).tril_(position)
        h = self.self_attn(h, keys, values, mask=mask, q_start=position)
        h = _maybe(self.self_post_norm, h)
        x = _maybe(self.self_res_norm, x * self.res_gain + h)

        h = _maybe(self.cross_pre_norm, x)
        h = self.cross_attn(h, *cross_kv, mask=cross_mask, q_start=position)
        h = _maybe(self.cross_post_norm, h)
        x = _maybe(self.cross_res_norm, x * self.res_gain + h)

        x = x * self.res_gain + self.ff(x)
        return _maybe(self.final_norm, x)


def _layer_has_final_norm(cfg, index, n_layers):
    if cfg.ln_positions == "postln":
        return True
    # Swin v2 adds a norm every 6 layers; it cannot be combined with scanned (identical) layers
    return cfg.ln_positions == "swinv2" and not cfg.use_scan and (index + 1) % 6 == 0 and index != n_layers - 1


class _Stack(nn.Module):
    """Embeddings, layer stack and final norm shared by the text encoder and the image decoder."""

    def __init__(self, cfg, vocab_size, length, n_layers, layer_class, use_final_ln, factory):
        super().__init__()
        dim = cfg.d_model
        operations = factory["operations"]
        dd = {"dtype": factory["dtype"], "device": factory["device"]}
        self.embed_scale = math.sqrt(dim) if cfg.scale_embedding else 1.0
        self.embed_tokens = operations.Embedding(vocab_size, dim, **dd)
        self.embed_positions = operations.Embedding(length, dim, **dd) if cfg.use_absolute_position_embeddings else None
        self.layernorm_embedding = _norm(cfg, dim, **factory)
        self.layers = nn.ModuleList(
            layer_class(cfg, _layer_has_final_norm(cfg, i, n_layers), factory) for i in range(n_layers))
        # postln already normalises the output of every layer
        self.final_ln = _norm(cfg, dim, **factory) if use_final_ln and cfg.ln_positions != "postln" else None

    def embed(self, input_ids, start=0):
        x = self.embed_tokens(input_ids) * self.embed_scale
        if self.embed_positions is not None:
            positions = torch.arange(start, start + input_ids.shape[1], device=input_ids.device)
            x = x + self.embed_positions(positions)
        return self.layernorm_embedding(x)


class TextEncoder(_Stack):
    def __init__(self, cfg, factory):
        super().__init__(cfg, cfg.encoder_vocab_size, cfg.max_text_length, cfg.encoder_layers,
                         EncoderLayer, cfg.use_final_ln_encoder, factory)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        mask = attention_mask.to(torch.bool)[:, None, None, :]
        for layer in self.layers:
            x = layer(x, mask)
        return _maybe(self.final_ln, x)


class ImageDecoder(_Stack):
    def __init__(self, cfg, factory):
        # one extra embedding for the start token
        super().__init__(cfg, cfg.image_vocab_size + 1, cfg.image_length, cfg.decoder_layers,
                         DecoderLayer, cfg.use_final_ln_decoder, factory)

    def cross_keys_values(self, encoder_states):
        """Per-layer cross-attention keys/values. They depend only on the text, so compute them once."""
        return [layer.cross_attn.keys_values(encoder_states) for layer in self.layers]

    def forward(self, input_ids, cross_kvs, cross_mask, caches=None, position=0):
        x = self.embed(input_ids, start=position)
        for i, layer in enumerate(self.layers):
            x = layer(x, cross_kvs[i], cross_mask, None if caches is None else caches[i], position)
        return _maybe(self.final_ln, x)


def sample_tokens(logits, uniform, top_k=50, top_p=1.0, temperature=1.0):
    """Sample one token per row. Filters are applied in the order transformers uses:
    temperature, then top-k, then top-p (nucleus).

    ``logits``: [B, V] float. ``uniform``: [B] samples from U[0, 1), which makes the draw
    reproducible independently of the device the model runs on.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    if temperature != 1.0:
        logits = logits / temperature
    vocab = logits.shape[-1]
    k = vocab if not top_k or top_k <= 0 else min(top_k, vocab)
    values, indices = logits.topk(k, dim=-1)  # sorted, most likely first
    probs = values.softmax(dim=-1)
    if top_p < 1.0:
        # keep the smallest set of most likely tokens whose probability mass reaches top_p
        keep = (probs.cumsum(dim=-1) - probs) < top_p
        keep[:, 0] = True
        probs = probs * keep
    cdf = probs.cumsum(dim=-1)
    target = uniform.to(cdf)[:, None] * cdf[:, -1:]
    choice = (cdf <= target).sum(dim=-1, keepdim=True).clamp_(max=k - 1)
    return indices.gather(-1, choice).squeeze(-1)


class DalleBart(nn.Module):
    def __init__(self, cfg: DalleBartConfig, dtype=None, device=None, operations=default_ops):
        super().__init__()
        self.config = cfg
        factory = {"dtype": dtype, "device": device, "operations": operations}
        self.encoder = TextEncoder(cfg, factory)
        self.decoder = ImageDecoder(cfg, factory)
        # Always float32: logits reach about 100, where half precision only resolves steps of 1/16 (fp16) or
        # 1/2 (bf16), and super-conditioning multiplies that rounding error by condition_scale.
        self.lm_head = operations.Linear(cfg.d_model, cfg.image_vocab_size + 1, bias=False, dtype=torch.float32, device=device)

    def encode_text(self, input_ids, attention_mask):
        """[B, T] token ids and mask -> [B, T, D] encoder states."""
        return self.encoder(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        """Teacher-forced pass: logits [B, T, image_vocab_size + 1] for every decoder position."""
        states = self.encode_text(input_ids, attention_mask)
        mask = attention_mask.to(torch.bool)[:, None, None, :]
        hidden = self.decoder(decoder_input_ids, self.decoder.cross_keys_values(states), mask)
        return self.lm_head(hidden.float())

    @torch.no_grad()
    def generate(self, cond, cond_mask, uncond=None, uncond_mask=None, *, uniform, condition_scale=10.0,
                 top_k=50, top_p=1.0, temperature=1.0, callback=None):
        """Autoregressively sample image tokens.

        ``cond`` / ``uncond``: [B, T, D] encoder states with their [B, T] masks. ``uncond`` drives
        super-conditioning: logits = uncond + condition_scale * (cond - uncond).
        ``uniform``: [B, image_length] samples from U[0, 1), one per token.
        ``callback(step, total)`` is called after each token.
        Returns [B, image_length] VQGAN code indices (int64, on the CPU).
        """
        cfg = self.config
        batch, length = cond.shape[0], cfg.image_length
        if tuple(uniform.shape) != (batch, length):
            raise ValueError(f"uniform must have shape {(batch, length)}, got {tuple(uniform.shape)}")
        guided = uncond is not None and condition_scale != 1.0
        if guided:
            # both branches share one batched decoder pass
            states = torch.cat([cond, uncond])
            mask = torch.cat([cond_mask, uncond_mask])
        else:
            states, mask = cond, cond_mask
        rows, device = states.shape[0], states.device
        mask = mask.to(torch.bool)[:, None, None, :]

        cross_kvs = self.decoder.cross_keys_values(states)
        heads, head_dim = cfg.decoder_attention_heads, cfg.d_model // cfg.decoder_attention_heads
        caches = [tuple(torch.empty(rows, heads, length, head_dim, dtype=states.dtype, device=device) for _ in range(2))
                  for _ in self.decoder.layers]

        uniform = uniform.to(device="cpu", dtype=torch.float32)
        tokens = torch.empty(batch, length, dtype=torch.long)
        current = torch.full((rows, 1), cfg.decoder_start_token_id, dtype=torch.long, device=device)
        for step in range(length):
            hidden = self.decoder(current, cross_kvs, mask, caches, position=step)
            logits = self.lm_head(hidden[:, -1].float())
            if guided:
                logits = logits[batch:] + condition_scale * (logits[:batch] - logits[batch:])
            # Sampling runs on the CPU: it is a tiny tensor, and it keeps the result independent of
            # how each GPU backend implements top-k and cumulative sums. Ids past the codebook
            # (start/pad tokens, vocabulary padding) are dropped so the VQGAN can always decode.
            next_tokens = sample_tokens(logits[:, :cfg.num_image_codes].cpu(), uniform[:, step],
                                        top_k=top_k, top_p=top_p, temperature=temperature)
            tokens[:, step] = next_tokens
            current = next_tokens.to(device).repeat(rows // batch)[:, None]
            if callback is not None:
                callback(step + 1, length)
        return tokens
