# Plan

State as of 2026-10-12, after the GPU run on the Mac. That run closed items 1 to 3 of the previous plan for Apple Silicon and changed the code in one place (see "Changed by the Mac run"). CUDA, DirectML and ROCm are still untested.

## Done and verified

### Raspberry Pi (CPU, float32)

ComfyUI 0.39.0 (`n1ckfg/ComfyUI` at `e8665ca`), torch 2.14, transformers 5.19.

| What | How it was checked | Result |
|---|---|---|
| Flax checkpoint reader (`flax_io.py`) | Parsed the real mini (1.75 GB) and VQGAN checkpoints | 437.8M and 76.1M parameters, parse in under 0.2 s |
| DalleBart, every architecture switch | 23 tiny random models, logits vs. the original Flax code | max abs diff about 1e-6 to 3e-5 |
| KV-cached decoding (the path `generate()` uses) | Same 23 models, step-by-step vs. full forward | max abs diff about 1e-6 |
| DalleBart with the published mini weights | Encoder states and teacher-forced logits at all 256 positions vs. Flax | 2.4e-5 and 1.8e-4 (logit magnitude 66) |
| VQGAN decoder | Decoded image vs. vqgan-jax | 3.2e-5 |
| Text normaliser and tokenizer | 24 prompts (emoji, mojibake, hashtags, HTML, truncation) vs. upstream | identical text and token ids |
| Sampling filters | Empirical frequencies vs. transformers' temperature / top-k / top-p warpers | within sampling noise |
| Real generation | "an armchair in the shape of an avocado", mini, seed 0, batch of 2, standalone | correct image; 292 s for 256 tokens |
| Real mini inside ComfyUI | Same prompt and seed, batch of 1, through the HTTP API | success in 333 s; same image as image 0 of the standalone batch (pixels differ by at most 1/255, from PNG rounding) |
| ComfyUI layer classes | Loaded through `comfy.ops.disable_weight_init`, normal and deferred-weight mode, and with every layer forced onto the weight-casting (low VRAM) path | same logits |
| Nodes in a running ComfyUI | Loader, Text Encode, Sampler (with and without negative), Decode, CLIP Rank, Save, through the HTTP API with a tiny model and the real VQGAN | all succeeded; VQGAN auto-download worked |
| CLIP Rank ordering | Known images against two prompts | sensible order and scores |
| Example workflow | Loaded in the real frontend with headless Chromium | 5 nodes, links and widget values intact |

### Mac (M2 Max, 64 GB, MPS)

ComfyUI 0.39.0 (same commit), torch 2.13.0.dev20260530, transformers 5.0.0rc3, macOS 15.7. ComfyUI was started with only this node pack loaded. Numbers are from the code as it is now, after the change below.

| What | How it was checked | Result |
|---|---|---|
| Example workflow on MPS | `workflows/dalle_mini_example.json` (mini, `auto` precision, batch of 4) through the HTTP API | fp16 chosen, `Requested to load DalleBart`, four plausible images in 4 s. No CPU fallback, warning or traceback in 32 prompts, 14 of them on the final code |
| Every precision on MPS | mini and mega in fp16, fp32 and bf16 | all load and give plausible images |
| MPS against CPU float32 | `tests/test_parity.py device`, mini and mega: teacher-forced logits, and the token super-conditioning would pick at each of 256 positions | fp32: logits within 1.8e-4 (magnitude 141), greedy tokens 256/256 identical. fp16: same pick at 99.2% (mini) and 98.8% (mega) of positions. bf16: 94.1% and 91.4% |
| Sampler settings on MPS | `condition_scale` 1, negative prompt, top-p with temperature, hashtag prompt, interrupt during a mega batch of 32 | all succeed. Interrupt stops within 0.2 s and the next prompt runs normally |
| A seed keeps its images when the batch grows | Tokens of image 1 from a batch of 1 and a batch of 4, 5 seeds, mini and mega, fp16 and fp32 | identical tokens in all 20 cases. The saved PNGs differ by 1/255 in 5 of 65536 pixels, from the VQGAN decoding a different batch |
| Low VRAM on a real GPU | Nodes run in-process with ComfyUI's VRAM state forced, because ComfyUI ignores `--lowvram` and `--novram` on MPS: `NO_VRAM` (all 681 mega layers cast from the CPU at use) and `LOW_VRAM` with a pretend 1.6 GB GPU (mega split 1.1 GB on MPS, 3.9 GB on the CPU) | 512/512 tokens identical to the full load, for mini and mega |
| Mega loads | Auto-download through the loader, then the nodes | 2588.0M parameters, loads in 2.3 s. `num_image_codes` is 16384 with `image_vocab_size` 16415. fp16: private memory peaks at 5.2 GB, resident memory at 15.1 GB counting the mapped checkpoint. fp32: 10.1 GB and 19.6 GB |
| Mega against upstream | `flax_reference.py pretrained --generate`, then `test_parity.py pretrained` | encoder states 3.0e-5, logits 4.0e-4 (magnitude 138), greedy `generate()` tokens 256/256 identical |
| Mini against upstream `generate()` | Same, for mini. This is the run that never finished on the Pi | 256/256 tokens identical. The upstream reference run took under 3 minutes for mini and 26 minutes for mega on this CPU |
| Earlier parity checks, again | 23 variants and the mini checks, on torch 2.13 and on torch 2.1.0 | all pass on both |
| Auto-download | mini, mega, VQGAN and the hashtag word list, each fetched by the nodes on first use | worked; mega (10.4 GB) took 4 minutes |
| CLIP Rank | transformers 5.0.0rc3 on MPS with ViT-B/32, ViT-B/16 and ViT-L/14, with and without `top_n` | sorted scores, the two avocado armchairs ranked above the plain armchair and the plain avocados |
| Older ComfyUI | v0.3.75 (2025-11): full server through the HTTP API, with CLIP Rank. v0.3.30 (2025-04): nodes run in-process against its `comfy` package, because its server does not start with the frontend packages installed here | both work through the `ModelPatcher` fallback |

Speed on MPS in fp16, 256 tokens with super-conditioning:

| | 1 image | 4 images |
|---|---|---|
| mini | 1.9 s | 2.6 s |
| mega | 5.0 s | 9.4 s |

The M2 Max CPU in fp32 needs 18 s (mini) and 96 s (mega) for 4 images.

## Changed by the Mac run

- **The output projection is always float32** (`DalleBart.lm_head`, `loading.py`). The plan suspected the `.float()` after `lm_head`, and that was the place. Logits reach about 100, where fp16 only resolves steps of 1/16 and bf16 steps of 1/2, and super-conditioning multiplies that rounding error by `condition_scale`. Nothing looked broken, but with mega in fp16 the most likely token matched fp32 at only 77.7% of positions (bf16: 30.5%; mini: 96.5% and 74.2%). Casting the last hidden state to float32 before the projection brings that to the figures in the table. It costs 64 MB for mega and 32 MB for mini; the weights do not need to be more precise, only the output does.
- **`tests/test_parity.py device`** compares fp32, fp16 and bf16 on a GPU with fp32 on the CPU. It needs no Flax reference and takes under a minute for mini, so it is the quickest check on a new backend.
- The docstring of `ops.py` named the wrong ComfyUI layer set.

## Not verified yet

### 1. CUDA, DirectML and ROCm

Only the CPU and MPS have run. On a machine with one of the others:

```sh
python tests/test_parity.py device --model ComfyUI/models/dalle_mini/mini   # add --device for anything but cuda
```

then load `workflows/dalle_mini_example.json` and run it. Still open from the previous plan:

- **Dynamic VRAM on NVIDIA** (`ModelPatcherDynamic`, enabled when comfy-aimdo is installed). comfy-aimdo only supports Windows and Linux, so it could not run here. In that mode `Linear` layers are built without a weight, which is why `loading.py` picks the float32 projection by name instead of reading the dtype off the module.
- **DirectML**: int64 token ids into `Embedding`, and `scaled_dot_product_attention`. `test_parity.py device` takes a device string, which DirectML does not have, so check it through the workflow.

### 2. Smaller gaps

- bf16 on MPS needs a torch newer than 2.1, which rejects the dtype there with a clear error. fp16 and fp32 pass on torch 2.1.0.
- ComfyUI v0.3.30 was checked in-process only, not through its server.

## Decisions for you

- **Add to Time-and-a-GAN** as a submodule, like the other ports: `git submodule add https://github.com/n1ckfg/ComfyUI-Dall-E-Mini`.
- **License.** I copied upstream's Apache 2.0 `LICENSE`, since the model and text code are derived from it.
- **`../dalle-mini/ARCHITECTURE.md` is wrong about mini** (the copy next to this repo on the Pi). Its comparison table says standard FFN, biases on, FFN dim 4096. The published `config.json` says GLU, NormFormer, no bias, FFN dim 2730, `use_scan: true`. The table's mini column describes `tools/train/config/mini/config.json`, which is not what was released.
- **Upstream bug worth knowing about:** `FlaxBartDecoder` never defines `final_ln` when `ln_positions` is `postln`/`deepnet` or `use_final_ln_decoder` is false, so those configurations crash upstream. The reference script patches it the way the encoder handles it; this port supports them.

## Possible follow-ups

- **Live preview** while sampling: decode the partial token grid every row and send it through `ProgressBar.update_absolute(preview=...)`. The image fills in top to bottom.
- **Image prompts**: port the VQGAN encoder (its weights are already in the checkpoint) and let the sampler start from the top rows of an existing image.
- **fp16 safetensors cache for mega**, to halve the 10.4 GB on disk. Load time is no longer a reason: the checkpoint loads in 2.3 s from an SSD.
- **CPU speed**: generation costs about 1.1 s per token on the Pi regardless of batch size, because every step reads all 850 MB of decoder weights. bf16 or int8 weights would help; nothing else will.
- A second example workflow with CLIP Rank and an upscaler.

## Test assets

### On the Mac

The models are where the loader put them, in `ComfyUI/models/dalle_mini/` (`mini` 1.75 GB, `mega` 10.4 GB, `vqgan_imagenet_f16_16384` 0.3 GB). CLIP Rank left three `models--openai--clip-vit-*` folders (2.8 GB) in `~/.cache/huggingface/hub/`.

Everything else is in `~/.cache/dalle-mini-port-test/` (1.1 GB):

| Path | Contents |
|---|---|
| `venv-flax/` | Python 3.11, JAX 0.4.20 / Flax 0.7.5 / transformers 4.25.1, for `tests/flax_reference.py` |
| `venv-torch21/` | torch 2.1.0, to check the oldest supported torch |
| `src/` | Shallow clones of `borisdayma/dalle-mini` and `patil-suraj/vqgan-jax` |
| `weights/` | Symlinks to `mini`, `mega` and the VQGAN in `ComfyUI/models/dalle_mini/` |
| `ref/` | Flax reference outputs: `variants/`, and `mini/` and `mega/` with upstream's `generate()` tokens |
| `ComfyUI-v0.3.30/`, `ComfyUI-v0.3.75/` | Shallow clones of older ComfyUI releases with this repo symlinked into `custom_nodes/` |

To rerun the checks:

```sh
W=~/.cache/dalle-mini-port-test
python tests/test_parity.py sampling
python tests/test_parity.py variants --ref $W/ref/variants
python tests/test_parity.py pretrained --ref $W/ref/mini --model $W/weights/mini --vqgan $W/weights/vqgan
python tests/test_parity.py pretrained --ref $W/ref/mega --model $W/weights/mega --vqgan $W/weights/vqgan
python tests/test_parity.py device --model $W/weights/mini
python tests/test_parity.py device --model $W/weights/mega
```

To dump the references again (the mega run needs about 16 GB of RAM):

```sh
cd $W
venv-flax/bin/python <repo>/tests/flax_reference.py variants --dalle-mini-src src/dalle-mini/src --out ref/variants
venv-flax/bin/python <repo>/tests/flax_reference.py pretrained --generate --dalle-mini-src src/dalle-mini/src \
    --vqgan-jax-src src/vqgan-jax --model weights/mini --vqgan weights/vqgan --out ref/mini
```

### On the Raspberry Pi

Everything heavy is in `~/.cache/dalle-mini-port-test/` (4.9 GB), outside the repo, because `/tmp` there is a 500 MB RAM disk:

| Path | Contents |
|---|---|
| `venv-comfy/`, `ComfyUI/` | CPU ComfyUI test install; this repo is symlinked into `custom_nodes/`, and `weights/mini` into `models/dalle_mini/` |
| `venv-flax/` | JAX 0.4.20 / Flax 0.7.5 / transformers 4.25.1, for `tests/flax_reference.py` |
| `weights/` | mini and VQGAN checkpoints |
| `ref/` | Flax reference outputs (`variants/`, `mini/`), without upstream's `generate()` tokens |
| `out/` | Generated test images |

The reference runs also left `models--dalle-mini--dalle-mini` (word list) and `models--openai--clip-vit-base-patch32` in `~/.cache/huggingface/hub/`. The checks there use `$W/venv-comfy/bin/python` with the same commands as above.

Delete either folder when you no longer need it.
