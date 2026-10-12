# Plan

State as of 2026-10-12. Nothing is committed yet; every file in the repo is new and untracked.

## Done and verified

All of this was run on the Raspberry Pi (CPU, float32) against ComfyUI 0.39.0 (`n1ckfg/ComfyUI` at `e8665ca`, torch 2.14, transformers 5.19).

| What | How it was checked | Result |
|---|---|---|
| Flax checkpoint reader (`flax_io.py`) | Parsed the real mini (1.75 GB) and VQGAN checkpoints | 437.8M and 76.1M parameters, parse in under 0.2 s |
| DalleBart, every architecture switch | 23 tiny random models, logits vs. the original Flax code | max abs diff about 1e-6 to 3e-5 |
| KV-cached decoding (the path `generate()` uses) | Same 23 models, step-by-step vs. full forward | max abs diff about 1e-6 |
| DalleBart with the published mini weights | Encoder states and teacher-forced logits at all 256 positions vs. Flax | 2.4e-5 and 1.8e-4 (logit magnitude 66) |
| VQGAN decoder | Decoded image vs. vqgan-jax | 3.2e-5 |
| Text normaliser and tokenizer | 24 prompts (emoji, mojibake, hashtags, HTML, truncation) vs. upstream | identical text and token ids |
| Sampling filters | Empirical frequencies vs. transformers' temperature / top-k / top-p warpers | within sampling noise |
| Real generation | "an armchair in the shape of an avocado", mini, seed 0, standalone | correct image; 292 s for 256 tokens |
| ComfyUI layer classes | Loaded through `comfy.ops.disable_weight_init`, normal and deferred-weight mode, and with every layer forced onto the weight-casting (low VRAM) path | same logits |
| Nodes in a running ComfyUI | Loader, Text Encode, Sampler (with and without negative), Decode, CLIP Rank, Save, through the HTTP API with a tiny model and the real VQGAN | all succeeded; VQGAN auto-download worked |
| CLIP Rank ordering | Known images against two prompts | sensible order and scores |
| Example workflow | Loaded in the real frontend with headless Chromium | 5 nodes, links and widget values intact |

## Not verified yet

In priority order. Items 1 to 3 need a machine with a GPU.

### 1. Run it on a GPU

Nothing has run on CUDA, MPS, DirectML or ROCm, and nothing has run in fp16. The code goes through ComfyUI's device management and uses only common operations (sampling itself is done on the CPU), but that is reasoning, not a test.

- Install into a GPU ComfyUI, `pip install -r requirements.txt`, load `workflows/dalle_mini_example.json`, run with `mini`.
- Check the console shows `Requested to load DalleBart` and the image is a plausible avocado armchair.
- Things most likely to need a fix:
  - **fp16 quality.** Upstream ran fp16 on GPU, so it should hold. If images look broken, set the loader's `precision` to `fp32` to confirm, then look at where the logits are cast (`DalleBart.generate`, the `.float()` after `lm_head`).
  - **Dynamic VRAM on NVIDIA** (`ModelPatcherDynamic`, enabled when comfy-aimdo is installed). Weight loading in that mode was simulated on CPU; the patcher itself falls back to the classic one on CPU, so it has not run.
  - **DirectML**: int64 token ids into `Embedding`, and `scaled_dot_product_attention`.
  - **MPS**: fp16 `LayerNorm` and attention.

### 2. Load mega

The 10.4 GB mega checkpoint was never downloaded (fp32 does not fit in this Pi's RAM). It uses the same architecture as mini with larger numbers, so the risk is low, but check:

- It loads and generates. Host RAM while loading should peak near 5.2 GB (fp16) plus page cache.
- `num_image_codes` comes out as 16384 even though mega's `image_vocab_size` is 16415.

### 3. Token-for-token comparison with upstream `generate()`

This is the run that was cancelled: upstream's JAX loop needed more than 25 minutes on this CPU without finishing. It is now opt-in.

```sh
# JAX environment (see the docstring of tests/flax_reference.py), ideally with a GPU build of jaxlib
python tests/flax_reference.py pretrained --generate \
    --dalle-mini-src ../dalle-mini/src --vqgan-jax-src <dir holding vqgan_jax/> \
    --model <mini folder> --vqgan <vqgan folder> --out ref/mini

# PyTorch environment
python tests/test_parity.py pretrained --ref ref/mini --model <mini folder> --vqgan <vqgan folder>
```

It replaces upstream's sampling with argmax and expects this port to produce the same 256 tokens at `temperature=0`. What it would add over the checks above is the super-conditioning arithmetic and the loop glue; the logits feeding them already match.

### 4. Smaller gaps

- Only ComfyUI 0.39 was tested. Older versions should work (`CoreModelPatcher` falls back to `ModelPatcher`), but `load_state_dict(assign=True)` needs torch 2.1 or newer.
- CLIP Rank was only run with transformers 5.19 and ViT-B/32.
- The `mini` and `mega` auto-download was not exercised (the same code downloaded the VQGAN; mini was linked in to save 1.75 GB).

## Decisions for you

- **Commit and push.** `origin` is `git@github.com:n1ckfg/ComfyUI-Dall-E-Mini`. `PLAN.md` and `tests/` are safe to publish; drop them if you prefer the repo lean.
- **Add to Time-and-a-GAN** as a submodule, like the other ports: `git submodule add https://github.com/n1ckfg/ComfyUI-Dall-E-Mini`.
- **License.** I copied upstream's Apache 2.0 `LICENSE`, since the model and text code are derived from it.
- **`../dalle-mini/ARCHITECTURE.md` is wrong about mini.** Its comparison table says standard FFN, biases on, FFN dim 4096. The published `config.json` says GLU, NormFormer, no bias, FFN dim 2730, `use_scan: true`. The table's mini column describes `tools/train/config/mini/config.json`, which is not what was released.
- **Upstream bug worth knowing about:** `FlaxBartDecoder` never defines `final_ln` when `ln_positions` is `postln`/`deepnet` or `use_final_ln_decoder` is false, so those configurations crash upstream. The reference script patches it the way the encoder handles it; this port supports them.

## Possible follow-ups

- **Live preview** while sampling: decode the partial token grid every row and send it through `ProgressBar.update_absolute(preview=...)`. The image fills in top to bottom.
- **Image prompts**: port the VQGAN encoder (its weights are already in the checkpoint) and let the sampler start from the top rows of an existing image.
- **fp16 safetensors cache for mega**, to halve the 10.4 GB on disk and the load time.
- **CPU speed**: generation costs about 1.1 s per token on the Pi regardless of batch size, because every step reads all 850 MB of decoder weights. bf16 or int8 weights would help; nothing else will.
- A second example workflow with CLIP Rank and an upscaler.

## Test assets on this machine

Everything heavy is in `~/.cache/dalle-mini-port-test/` (4.9 GB), outside the repo, because `/tmp` here is a 500 MB RAM disk:

| Path | Contents |
|---|---|
| `venv-comfy/`, `ComfyUI/` | CPU ComfyUI test install; this repo is symlinked into `custom_nodes/` |
| `venv-flax/` | JAX 0.4.20 / Flax 0.7.5 / transformers 4.25.1, for `tests/flax_reference.py` |
| `weights/` | mini and VQGAN checkpoints |
| `ref/` | Flax reference outputs (`variants/`, `mini/`) |
| `out/` | Generated test images |

Delete the folder when you no longer need it. The reference runs also left `models--dalle-mini--dalle-mini` (word list) and `models--openai--clip-vit-base-patch32` in `~/.cache/huggingface/hub/`.

To rerun the checks here:

```sh
W=~/.cache/dalle-mini-port-test
$W/venv-comfy/bin/python tests/test_parity.py sampling
$W/venv-comfy/bin/python tests/test_parity.py variants --ref $W/ref/variants
$W/venv-comfy/bin/python tests/test_parity.py pretrained --ref $W/ref/mini --model $W/weights/mini --vqgan $W/weights/vqgan
```
