# ComfyUI-Dall-E-Mini

[DALL·E Mini / Mega](https://github.com/borisdayma/dalle-mini) text-to-image as native ComfyUI nodes.

The original is JAX/Flax code written for TPU pods. This is a PyTorch reimplementation of its inference path that reads the official checkpoints directly, so it runs wherever ComfyUI runs: NVIDIA (CUDA), Apple Silicon (MPS), AMD/Intel on Windows (DirectML), AMD on Linux (ROCm) and CPU. No JAX, Flax or conversion step.

## Install

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/n1ckfg/ComfyUI-Dall-E-Mini
pip install -r ComfyUI-Dall-E-Mini/requirements.txt
```

Then load `workflows/dalle_mini_example.json`.

## Models

Models download from Hugging Face on first use into `ComfyUI/models/dalle_mini/`:

| Folder | Source | Download | Weights in memory |
|---|---|---|---|
| `mini/` | [dalle-mini/dalle-mini](https://huggingface.co/dalle-mini/dalle-mini) | 1.75 GB | 0.9 GB fp16, 1.75 GB fp32 |
| `mega/` | [dalle-mini/dalle-mega](https://huggingface.co/dalle-mini/dalle-mega) | 10.4 GB | 5.2 GB fp16, 10.4 GB fp32 |
| `vqgan_imagenet_f16_16384/` | [dalle-mini/vqgan_imagenet_f16_16384](https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384) | 0.3 GB | 0.2 GB |

To install by hand, put `config.json`, `tokenizer.json` and `flax_model.msgpack` from the model repo into the folder (the VQGAN needs only `config.json` and `flax_model.msgpack`). Any other folder with those three files, such as a fine-tune, shows up in the loader too.

The `.msgpack` files are read as plain data through a memory map; nothing is unpickled.

## Nodes

All under **DALL-E Mini**.

| Node | In | Out |
|---|---|---|
| **DALL·E Mini Loader** | model, precision | `dalle_model`, `vqgan` |
| **DALL·E Mini Text Encode** | `dalle_model`, text | `conditioning` |
| **DALL·E Mini Sampler** | `dalle_model`, positive, optional negative, sampling settings | `tokens` |
| **DALL·E Mini VQGAN Decode** | `vqgan`, `tokens` | `IMAGE` |
| **DALL·E Mini CLIP Rank** | `IMAGE`, text | `IMAGE` sorted best first, scores |

Images are 256×256; chain any ComfyUI upscaler after the decode node.

### Sampler settings

| Setting | Default | |
|---|---|---|
| `seed` | | Image *i* of a batch depends only on the seed and *i*, so raising `batch_size` keeps the earlier images. A different `precision` or device can give a different image for the same seed: one changed token changes every token after it. |
| `batch_size` | 1 | Number of images. DALL·E Mini is at its best when you make several and pick. |
| `condition_scale` | 10 | Super-conditioning: classifier-free guidance on the token logits. Higher follows the prompt more closely. 1 turns it off and doubles the speed. |
| `top_k` | 50 | Sample each token from the k most likely codes. 0 considers all 16384. |
| `top_p` | 1.0 | Nucleus sampling. 1.0 is off. |
| `temperature` | 1.0 | 0 always takes the most likely code. |
| `negative` | empty prompt | What super-conditioning steers away from. |

The defaults are the ones the upstream inference notebook ends up with.

**CLIP Rank** is the optional re-ranking step from the upstream pipeline. It downloads the chosen CLIP model (about 0.6 GB for ViT-B/32) through `transformers` on first use.

## How it works

DALL·E Mini is not a diffusion model. It treats text-to-image as translation:

1. The prompt is normalised and BPE-tokenized (at most 64 tokens).
2. A BART-style encoder reads the text. A causal decoder then samples 256 tokens one at a time, each an index into a 16384-entry VQGAN codebook, one per cell of a 16×16 grid.
3. The VQGAN decoder turns the token grid into a 256×256 image.

`dalle_mini_torch/` holds the model code and has no ComfyUI imports; `nodes.py` wires it into ComfyUI.

## Differences from upstream

- **Weights** are read straight from `flax_model.msgpack`, including checkpoints saved with scanned (stacked) layers.
- **Generation** uses a key/value cache and runs the conditional and unconditional branches as one batch. Upstream ran two decoder passes per token and recomputed cross-attention keys each step.
- **Logits are float32 in every precision.** Upstream computes them in the model's dtype. They reach about 100, where fp16 only resolves steps of 1/16, and super-conditioning multiplies that rounding error by the condition scale.
- **Negative prompt**: the unconditional branch of super-conditioning accepts any prompt, not only the empty one.
- **Tokens are restricted to the 16384 real codebook entries.** The model's output layer also covers start/padding ids that the VQGAN cannot decode.
- **An empty prompt is the trained unconditional input.** Upstream would normalise it to a single space, which is a different token sequence from the blank captions used in training.
- **Hashtag splitting** loads its 34 MB word list only when a prompt contains a `#hashtag`.
- **Memory** is managed by ComfyUI (`ModelPatcher`), so the models are offloaded and partially loaded like any other.
- Every architecture switch that upstream implements in `DalleBartConfig` is ported and checked against it, not only the two published configurations.
- Training, the TPU sharding code and the W&B integration are not ported.

## Status

Checked against the original JAX/Flax code on CPU (float32):

- 23 tiny random models covering every architecture switch match upstream logits to about 1e-6, and the cached decoding path matches the full forward pass.
- With the published mini and mega weights, text normalisation, token ids, encoder states, logits at all 256 positions and the VQGAN image match upstream.
- With sampling replaced by argmax, this port and upstream's `generate()` produce the same 256 tokens, for mini and for mega.

Run end to end in ComfyUI 0.39 on CPU (Raspberry Pi) and on Apple Silicon (MPS, M2 Max). The nodes also run against ComfyUI 0.3.75 and 0.3.30. On MPS:

- mini and mega work in fp16, fp32 and bf16. fp32 on MPS gives the same greedy tokens as the CPU. Against CPU fp32, super-conditioning picks the same most likely token at 99% of positions in fp16 and at 91% to 94% in bf16.
- 256 tokens take about 2 s with mini and 5 s with mega on MPS in fp16; four images take 3 s and 9 s.
- ComfyUI's low VRAM path, with weights cast from the CPU at use, gives the same tokens as a full load.

Not yet run: CUDA, DirectML, ROCm, and ComfyUI's dynamic VRAM mode on NVIDIA. See [PLAN.md](PLAN.md).

## Tests

`tests/flax_reference.py` runs the original JAX/Flax code and dumps its outputs; `tests/test_parity.py` compares this port against them and needs no JAX.

```sh
python tests/test_parity.py sampling
python tests/test_parity.py variants --ref <ref>/variants
python tests/test_parity.py pretrained --ref <ref>/mini --model <mini folder> --vqgan <vqgan folder>
```

`device` needs no reference data. It compares fp32, fp16 and bf16 on a GPU with fp32 on the CPU, which is the quickest check on a new backend:

```sh
python tests/test_parity.py device --model <mini folder> [--device cuda]
```

## Credits

DALL·E Mini is by Boris Dayma, Suraj Patil, Pedro Cuenca, Khalid Saifullah, Tanishq Abraham, Phúc Lê Khắc, Luke Melas and Ritobrata Ghosh. The VQGAN is from [taming-transformers](https://github.com/CompVis/taming-transformers) via [vqgan-jax](https://github.com/patil-suraj/vqgan-jax). Super-conditioning is by Katherine Crowson.

This port is derived from their code and is released under the same Apache 2.0 license.
