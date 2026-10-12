"""Dump ground-truth outputs from the original JAX/Flax implementation.

This is the only file that needs JAX. It runs the unmodified upstream code
(https://github.com/borisdayma/dalle-mini and https://github.com/patil-suraj/vqgan-jax)
and writes reference data that ``test_parity.py`` compares the PyTorch port against.

Environment used (Python 3.11):
    pip install jax==0.4.20 jaxlib==0.4.20 flax==0.7.5 transformers==4.25.1 "numpy<2" "scipy<1.13" \
        einops unidecode ftfy emoji pillow

    python tests/flax_reference.py variants --dalle-mini-src ../dalle-mini/src --out ref/variants
    python tests/flax_reference.py pretrained --dalle-mini-src ../dalle-mini/src --vqgan-jax-src <dir holding vqgan_jax/> \
        --model <mini dir> --vqgan <vqgan dir> --out ref/mini [--generate]

``--generate`` adds a token-for-token comparison with upstream's generate() loop. It needs a GPU to be
practical; everything else runs in a few minutes on a CPU.
"""

import argparse
import json
import os
import sys
import types

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np

# Names that jax 0.4 removed or moved but upstream still imports (type hints and TPU sharding rules).
if not hasattr(jnp, "DeviceArray"):
    jnp.DeviceArray = jax.Array
if not hasattr(jax.experimental, "PartitionSpec"):
    jax.experimental.PartitionSpec = jax.sharding.PartitionSpec
sys.modules.setdefault("wandb", types.ModuleType("wandb"))  # upstream imports it for W&B artifact downloads only

PROMPTS = [
    "an armchair in the shape of an avocado",
    "sunset over a lake in the mountains",
    "The Eiffel tower landing on the moon!!!",
    "Dali painting of WALL·E — 3,000 robots & a café, it's 50% off",
    "<b>A photo</b> of a #sunsetlake https://example.com/x.jpg \U0001F680",
    "",
]

# Tiny random models covering every architecture switch of DalleBartConfig.
TINY = dict(
    encoder_vocab_size=97, image_vocab_size=53, image_length=16, max_text_length=8, d_model=32,
    encoder_layers=2, encoder_ffn_dim=48, encoder_attention_heads=4,
    decoder_layers=3, decoder_ffn_dim=40, decoder_attention_heads=2,
    gradient_checkpointing=False,
)
VARIANTS = {
    "default": {},
    "unscanned": dict(use_scan=False),
    "ffn_bias": dict(use_glu=False, use_bias=True),
    "rmsnorm": dict(ln_type="rmsnorm"),
    "rmsnorm_force_scale": dict(ln_type="rmsnorm", force_ln_scale=True),
    "force_ln_scale": dict(force_ln_scale=True),
    "preln": dict(ln_positions="preln"),
    "postln": dict(ln_positions="postln"),
    "postln_unscanned": dict(ln_positions="postln", use_scan=False),
    "cogview": dict(ln_positions="cogview"),
    "subln": dict(ln_positions="subln", use_subln_init=True),
    "swinv2": dict(ln_positions="swinv2", encoder_layers=7, decoder_layers=13),
    "head_scale": dict(use_head_scale=True),
    "cosine_attention": dict(use_cosine_attention=True),
    "swin_positions": dict(use_swin_position_embeddings=True, use_absolute_position_embeddings=False),
    "sinkhorn": dict(sinkhorn_iters=3),
    "deepnet": dict(ln_positions="deepnet", use_deepnet_scaling=True),
    "no_final_ln": dict(use_final_ln_encoder=False, use_final_ln_decoder=False),
    "scale_embedding": dict(scale_embedding=True),
    "gelu_new": dict(activation_function="gelu_new"),
    "relu_ffn": dict(activation_function="relu", use_glu=False),
    "swish": dict(activation_function="swish"),
    "smelu": dict(activation_function="smelu"),
}


def _forward(module, params, input_ids, attention_mask, decoder_input_ids):
    batch, text_len = input_ids.shape
    image_len = decoder_input_ids.shape[1]
    return module.apply(
        {"params": params},
        jnp.asarray(input_ids, dtype="i4"),
        jnp.asarray(attention_mask, dtype="i4"),
        jnp.asarray(decoder_input_ids, dtype="i4"),
        jnp.ones((batch, image_len), dtype="i4"),
        jnp.broadcast_to(jnp.arange(text_len)[None], (batch, text_len)),
        jnp.broadcast_to(jnp.arange(image_len)[None], (batch, image_len)),
        return_dict=False,
    )[0]


def dump_variants(args):
    from flax.serialization import to_bytes
    from flax.traverse_util import flatten_dict, unflatten_dict

    from dalle_mini.model import modeling
    from dalle_mini.model.configuration import DalleBartConfig
    from dalle_mini.model.modeling import FlaxBartForConditionalGenerationModule

    # Upstream bug: FlaxBartDecoder.setup only assigns final_ln when the final norm exists, then reads it
    # unconditionally, so "postln"/"deepnet" and use_final_ln_decoder=False crash. The encoder falls back
    # to None; give the decoder the same fallback so those variants can be compared at all.
    modeling.FlaxBartDecoder.final_ln = None

    for index, (name, overrides) in enumerate(VARIANTS.items()):
        config = DalleBartConfig(**{**TINY, **overrides})
        module = FlaxBartForConditionalGenerationModule(config=config, dtype=jnp.float32)
        rng = np.random.default_rng(index)
        batch = 2
        input_ids = rng.integers(0, config.encoder_vocab_size, (batch, config.max_text_length))
        attention_mask = np.ones((batch, config.max_text_length), dtype=np.int64)
        attention_mask[1, 5:] = 0  # one padded prompt
        decoder_input_ids = rng.integers(0, config.image_vocab_size + 1, (batch, config.image_length))

        params = module.init(
            jax.random.PRNGKey(index),
            jnp.asarray(input_ids, dtype="i4"), jnp.asarray(attention_mask, dtype="i4"),
            jnp.asarray(decoder_input_ids, dtype="i4"), jnp.ones_like(decoder_input_ids, dtype="i4"),
            jnp.broadcast_to(jnp.arange(config.max_text_length)[None], input_ids.shape),
            jnp.broadcast_to(jnp.arange(config.image_length)[None], decoder_input_ids.shape),
        )["params"]
        # Fresh norms are all ones/zeros, which would hide a swapped scale or bias: randomise everything.
        flat = flatten_dict(jax.tree_util.tree_map(np.asarray, params))
        flat = {k: (v + rng.normal(0.0, 0.2, v.shape)).astype(np.float32) for k, v in flat.items()}
        params = unflatten_dict(flat)

        logits = _forward(module, params, input_ids, attention_mask, decoder_input_ids)

        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        config.save_pretrained(out)
        with open(os.path.join(out, "flax_model.msgpack"), "wb") as f:
            f.write(to_bytes(params))
        np.savez(os.path.join(out, "reference.npz"), input_ids=input_ids, attention_mask=attention_mask,
                 decoder_input_ids=decoder_input_ids, logits=np.asarray(logits))
        print(f"{name}: {len(flat)} arrays, logits std {float(np.std(logits)):.3f}")


def dump_pretrained(args):
    from dalle_mini import DalleBart, DalleBartProcessor

    sys.path.insert(0, args.vqgan_jax_src)
    from vqgan_jax.modeling_flax_vqgan import VQModel

    out = args.out
    os.makedirs(out, exist_ok=True)
    reference = {}

    def save(stage):
        np.savez(os.path.join(out, "reference.npz"), **reference)
        print(f"saved after {stage}:", sorted(reference), flush=True)

    # 1. text normalisation and tokenisation
    processor = DalleBartProcessor.from_pretrained(args.model)
    normalized = [processor.text_processor(p) if processor.normalize_text else p for p in PROMPTS]
    tokenized = processor(PROMPTS)
    reference.update({k: np.asarray(tokenized[k]) for k in
                      ("input_ids", "attention_mask", "input_ids_uncond", "attention_mask_uncond")})
    with open(os.path.join(out, "text.json"), "w", encoding="utf-8") as f:
        json.dump({"prompts": PROMPTS, "normalized": normalized}, f, ensure_ascii=False, indent=1)

    # 2. encoder states and teacher-forced logits from the real weights
    model, params = DalleBart.from_pretrained(args.model, dtype=jnp.float32, _do_init=False)
    config = model.config
    rng = np.random.default_rng(0)
    num_codes = min(config.image_vocab_size, config.decoder_start_token_id)
    decoder_input_ids = rng.integers(0, num_codes, (1, config.image_length))
    decoder_input_ids[:, 0] = config.decoder_start_token_id
    ids, mask = reference["input_ids"][:1], reference["attention_mask"][:1]
    encoder_states = model.encode(jnp.asarray(ids), jnp.asarray(mask), params=params)[0]
    logits = model(jnp.asarray(ids), jnp.asarray(mask), jnp.asarray(decoder_input_ids), params=params)[0]
    reference.update(decoder_input_ids=decoder_input_ids, encoder_states=np.asarray(encoder_states),
                     logits=np.asarray(logits))
    save("DalleBart forward")

    # 3. VQGAN decoding of random tokens
    vqgan, vqgan_params = VQModel.from_pretrained(args.vqgan, _do_init=False)
    vqgan_tokens = rng.integers(0, vqgan.config.n_embed, (1, config.image_length))
    reference.update(vqgan_tokens=vqgan_tokens,
                     vqgan_images=np.asarray(vqgan.decode_code(jnp.asarray(vqgan_tokens), params=vqgan_params)))
    save("VQGAN decode")

    # 4. the upstream generate() loop, made deterministic by replacing sampling with argmax.
    #    Slow without a GPU (tens of minutes for mini on a small CPU), hence opt-in.
    if args.generate:
        jax.random.categorical = lambda key, logits, axis=-1, shape=None: jnp.argmax(logits, axis=axis)
        generated = model.generate(
            jnp.asarray(ids), jnp.asarray(mask), params=params, prng_key=jax.random.PRNGKey(0),
            condition_scale=args.condition_scale,
            input_ids_uncond=jnp.asarray(reference["input_ids_uncond"][:1]),
            attention_mask_uncond=jnp.asarray(reference["attention_mask_uncond"][:1]),
        ).sequences
        reference.update(greedy_tokens=np.asarray(generated)[:, 1:],  # drop the start token
                         condition_scale=np.float32(args.condition_scale))
        save("generate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["variants", "pretrained"])
    parser.add_argument("--dalle-mini-src", required=True, help="path to dalle-mini/src")
    parser.add_argument("--vqgan-jax-src", help="directory containing the vqgan_jax package")
    parser.add_argument("--model", help="folder with the DALL·E Mini config.json, tokenizer and flax_model.msgpack")
    parser.add_argument("--vqgan", help="folder with the VQGAN config.json and flax_model.msgpack")
    parser.add_argument("--generate", action="store_true", help="also run upstream generate() (slow on CPU)")
    parser.add_argument("--condition-scale", type=float, default=10.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.dalle_mini_src)
    if args.mode == "variants":
        dump_variants(args)
    else:
        dump_pretrained(args)
