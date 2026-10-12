"""Compare the PyTorch port against reference data dumped by ``flax_reference.py``.

Needs only torch, numpy and tokenizers (plus transformers for the sampling check).

    python tests/test_parity.py sampling
    python tests/test_parity.py variants --ref ref/variants
    python tests/test_parity.py pretrained --ref ref/mini --model <mini dir> --vqgan <vqgan dir>
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle_mini_torch import load_dalle_bart, load_vqgan_decoder, sample_tokens  # noqa: E402
from dalle_mini_torch.text import TextProcessor  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


def close(name, actual, expected, tol):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        return check(name, False, f"shape {actual.shape} != {expected.shape}")
    diff = float(np.abs(actual - expected).max())
    scale = float(np.abs(expected).max())
    check(name, diff <= tol * max(scale, 1.0), f"max abs diff {diff:.2e} (reference magnitude {scale:.2e})")


def cached_logits(model, input_ids, attention_mask, decoder_input_ids):
    """Logits from the incremental, key/value-cached decoder path that generate() uses."""
    cfg = model.config
    states = model.encode_text(input_ids, attention_mask)
    mask = attention_mask.to(torch.bool)[:, None, None, :]
    cross = model.decoder.cross_keys_values(states)
    rows, heads = states.shape[0], cfg.decoder_attention_heads
    caches = [tuple(torch.zeros(rows, heads, cfg.image_length, cfg.d_model // heads) for _ in range(2))
              for _ in model.decoder.layers]
    steps = [model.lm_head(model.decoder(decoder_input_ids[:, t:t + 1], cross, mask, caches, position=t))
             for t in range(decoder_input_ids.shape[1])]
    return torch.cat(steps, dim=1)


@torch.no_grad()
def test_variants(args):
    for name in sorted(os.listdir(args.ref)):
        folder = os.path.join(args.ref, name)
        ref = np.load(os.path.join(folder, "reference.npz"))
        model = load_dalle_bart(folder)
        inputs = [torch.from_numpy(ref[k]) for k in ("input_ids", "attention_mask", "decoder_input_ids")]
        logits = model(*inputs)
        close(f"{name}: full forward vs Flax", logits, ref["logits"], 2e-4)
        close(f"{name}: cached decoding vs full forward", cached_logits(model, *inputs), logits, 2e-4)


@torch.no_grad()
def test_pretrained(args):
    ref = np.load(os.path.join(args.ref, "reference.npz"))
    with open(os.path.join(args.ref, "text.json"), encoding="utf-8") as f:
        text = json.load(f)

    model = load_dalle_bart(args.model)
    processor = TextProcessor(os.path.join(args.model, "tokenizer.json"), model.config,
                              os.path.join(args.model, "enwiki-words-frequency.txt"))
    print("text")
    for i, prompt in enumerate(text["prompts"]):
        ids, mask = processor(prompt)
        if prompt.strip():
            check(f"normalised {prompt!r}", processor.normalize(prompt) == text["normalized"][i],
                  f"-> {processor.normalize(prompt)!r}")
            check("  token ids", np.array_equal(ids.numpy(), ref["input_ids"][i:i + 1])
                  and np.array_equal(mask.numpy(), ref["attention_mask"][i:i + 1]))
        else:
            # deliberate difference: an empty prompt is tokenised as the trained blank caption
            check("empty prompt == upstream unconditional input",
                  np.array_equal(ids.numpy(), ref["input_ids_uncond"][:1])
                  and np.array_equal(mask.numpy(), ref["attention_mask_uncond"][:1]))

    print("DalleBart")
    ids, mask = processor(text["prompts"][0])
    states = model.encode_text(ids, mask)
    close("encoder states", states, ref["encoder_states"], 2e-4)
    decoder_input_ids = torch.from_numpy(ref["decoder_input_ids"])
    close("teacher-forced logits", model(ids, mask, decoder_input_ids), ref["logits"], 2e-4)

    print("generation")
    uncond_ids, uncond_mask = processor("")
    uncond = model.encode_text(uncond_ids, uncond_mask)
    if "greedy_tokens" in ref:
        tokens = model.generate(states, mask, uncond, uncond_mask, uniform=torch.zeros(1, model.config.image_length),
                                condition_scale=float(ref["condition_scale"]), temperature=0.0)
        same = int((tokens.numpy() == ref["greedy_tokens"]).sum())
        check("greedy super-conditioned tokens == upstream generate()", same == tokens.numel(),
              f"{same}/{tokens.numel()} tokens identical")
    else:
        print("  skip upstream generate() comparison (reference was dumped without --generate)")
    del model

    print("VQGAN")
    vqgan = load_vqgan_decoder(args.vqgan)
    images = vqgan.decode_tokens(torch.from_numpy(ref["vqgan_tokens"]).long())
    close("decoded image", images, ref["vqgan_images"].clip(0.0, 1.0), 2e-4)
    unclipped = ref["vqgan_images"]
    print(f"  reference decoder output range before clipping: {unclipped.min():.2f} .. {unclipped.max():.2f}")


def test_sampling(args):
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)

    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 200, generator=generator) * 3
    uniform = torch.rand(20000, generator=generator)
    for top_k, top_p, temperature in [(50, 1.0, 1.0), (0, 1.0, 1.0), (0, 0.9, 1.0), (40, 0.5, 0.7), (1, 1.0, 1.0),
                                      (0, 0.01, 2.0), (500, 0.99, 1.3)]:
        expected = logits
        if temperature != 1.0:
            expected = TemperatureLogitsWarper(temperature)(None, expected)
        if top_k:
            expected = TopKLogitsWarper(min(top_k, logits.shape[-1]))(None, expected)
        if top_p < 1.0:
            expected = TopPLogitsWarper(top_p)(None, expected)
        expected = expected.softmax(dim=-1)
        worst = 0.0
        for row in range(logits.shape[0]):
            drawn = sample_tokens(logits[row].expand(len(uniform), -1), uniform, top_k, top_p, temperature)
            freq = torch.bincount(drawn, minlength=logits.shape[-1]) / len(uniform)
            if bool((freq[expected[row] == 0] > 0).any()):
                worst = float("inf")  # sampled a token the filters should have removed
            worst = max(worst, float((freq - expected[row]).abs().max()))
        check(f"top_k={top_k} top_p={top_p} temperature={temperature}", worst < 0.02, f"max frequency error {worst:.4f}")
    greedy = sample_tokens(logits, torch.rand(4, generator=generator), temperature=0.0)
    check("temperature 0 is argmax", bool((greedy == logits.argmax(-1)).all()))
    edges = sample_tokens(logits[:2], torch.tensor([0.0, 1.0 - 1e-7]), top_k=0)
    check("extreme uniform draws stay in range", bool(((edges >= 0) & (edges < logits.shape[-1])).all()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["sampling", "variants", "pretrained"])
    parser.add_argument("--ref", help="reference folder written by flax_reference.py")
    parser.add_argument("--model", help="folder with the DALL·E Mini config.json, tokenizer and flax_model.msgpack")
    parser.add_argument("--vqgan", help="folder with the VQGAN config.json and flax_model.msgpack")
    args = parser.parse_args()
    {"sampling": test_sampling, "variants": test_variants, "pretrained": test_pretrained}[args.mode](args)
    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)
