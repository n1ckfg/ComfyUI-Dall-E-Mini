import logging
import os

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths

from .dalle_mini_torch import load_dalle_bart, load_vqgan_decoder
from .dalle_mini_torch.loading import CONFIG_NAME, WEIGHTS_NAME
from .dalle_mini_torch.text import TextProcessor

CATEGORY = "DALL-E Mini"
MODEL_FOLDER = "dalle_mini"
TOKENIZER_NAME = "tokenizer.json"
WORD_LIST_NAME = "enwiki-words-frequency.txt"

folder_paths.add_model_folder_path(MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER))

# name -> (Hugging Face repo, revision). Fetched into ComfyUI/models/dalle_mini/<name>/ on first use.
DALLE_REPOS = {
    "mini": ("dalle-mini/dalle-mini", None),  # 0.4B parameters, 1.75 GB
    "mega": ("dalle-mini/dalle-mega", None),  # 2.6B parameters, 10.4 GB
}
VQGAN_NAME = "vqgan_imagenet_f16_16384"
VQGAN_REPO = ("dalle-mini/vqgan_imagenet_f16_16384", "e93a26e7707683d349bf5d5c41c5b0ef69b677a9")  # 0.3 GB

CLIP_MODELS = ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# VQGAN activations for one 256x256 image peak at a few hundred MB, so decode a few at a time
DECODE_CHUNK = 4


def _has_files(folder, filenames):
    return all(os.path.isfile(os.path.join(folder, name)) for name in filenames)


def _local_dalle_models():
    names = []
    for base in folder_paths.get_folder_paths(MODEL_FOLDER):
        if os.path.isdir(base):
            names += [name for name in sorted(os.listdir(base))
                      if _has_files(os.path.join(base, name), (CONFIG_NAME, WEIGHTS_NAME, TOKENIZER_NAME))]
    return names


def _download(repo, revision, filenames, folder):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            f"huggingface_hub is needed to download {repo}. Install it, or download "
            f"{', '.join(filenames)} from https://huggingface.co/{repo} into {folder}") from e
    os.makedirs(folder, exist_ok=True)
    for name in filenames:
        if not os.path.isfile(os.path.join(folder, name)):
            logging.info(f"[DALL-E Mini] downloading {repo}/{name} to {folder}")
            hf_hub_download(repo_id=repo, filename=name, revision=revision, local_dir=folder)


def _model_folder(name, filenames, repo=None):
    """Find ``name`` in any registered dalle_mini model folder, downloading it if it is a known model."""
    bases = folder_paths.get_folder_paths(MODEL_FOLDER)
    for base in bases:
        if _has_files(os.path.join(base, name), filenames):
            return os.path.join(base, name)
    if repo is None:
        raise FileNotFoundError(f"DALL-E Mini model '{name}' not found. Expected a folder named '{name}' holding "
                                f"{', '.join(filenames)} in one of: {', '.join(bases)}")
    folder = os.path.join(bases[0], name)
    _download(repo[0], repo[1], filenames, folder)
    return folder


def _word_list(folder):
    """The hashtag-splitting word list ships with the DALL-E models; only prompts with a #hashtag need it."""
    if not os.path.isfile(os.path.join(folder, WORD_LIST_NAME)):
        _download(DALLE_REPOS["mini"][0], DALLE_REPOS["mini"][1], [WORD_LIST_NAME], folder)
    return os.path.join(folder, WORD_LIST_NAME)


def _pick_dtype(precision, device, weights_path):
    if mm.is_device_cpu(device):
        if precision not in ("auto", "fp32"):
            logging.info(f"[DALL-E Mini] {precision} is not practical on the CPU, using fp32")
        return torch.float32
    if precision == "auto":
        params = os.path.getsize(weights_path) // 4
        return torch.float16 if mm.should_use_fp16(device, model_params=params) else torch.float32
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def _patcher(model, load_device, offload_device):
    # CoreModelPatcher is ComfyUI's dynamic-VRAM-aware patcher; older versions only have ModelPatcher
    patcher_class = getattr(comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher)
    return patcher_class(model, load_device=load_device, offload_device=offload_device)


class DalleModel:
    def __init__(self, folder, precision):
        self.load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        self.dtype = _pick_dtype(precision, self.load_device, os.path.join(folder, WEIGHTS_NAME))
        self.model = load_dalle_bart(folder, dtype=self.dtype, device=offload_device,
                                     operations=comfy.ops.disable_weight_init)
        self.config = self.model.config
        self.processor = TextProcessor(os.path.join(folder, TOKENIZER_NAME), self.config, lambda: _word_list(folder))
        self.patcher = _patcher(self.model, self.load_device, offload_device)
        self._unconditional = None

    def encode(self, text):
        input_ids, mask = self.processor(text)
        mm.load_models_gpu([self.patcher])
        states = self.model.encode_text(input_ids.to(self.load_device), mask.to(self.load_device))
        return {"states": states.to(mm.intermediate_device()), "mask": mask}

    def unconditional(self):
        """Encoding of the empty prompt, the default negative for super-conditioning."""
        if self._unconditional is None:
            self._unconditional = self.encode("")
        return self._unconditional

    def prepare(self, conditioning, batch_size):
        """Move a conditioning to the sampling device and repeat it over the batch."""
        states, mask = conditioning["states"], conditioning["mask"]
        if states.shape[-1] != self.config.d_model or states.shape[1] != self.config.max_text_length:
            raise ValueError("This conditioning was encoded with a different DALL-E Mini model than the sampler uses.")
        if states.shape[0] not in (1, batch_size):
            raise ValueError(f"Conditioning holds {states.shape[0]} prompts, which does not match batch_size {batch_size}.")
        states = states.to(device=self.load_device, dtype=self.dtype).expand(batch_size, -1, -1)
        return states, mask.to(self.load_device).expand(batch_size, -1)


class DalleVQGAN:
    def __init__(self, folder):
        self.load_device = mm.vae_device()
        offload_device = mm.vae_offload_device()
        # 76M parameters at 256x256: fp32 costs nothing and avoids half-precision artifacts in the decoder
        self.model = load_vqgan_decoder(folder, dtype=torch.float32, device=offload_device,
                                        operations=comfy.ops.disable_weight_init)
        self.patcher = _patcher(self.model, self.load_device, offload_device)

    def decode(self, tokens):
        mm.load_models_gpu([self.patcher])
        images = [self.model.decode_tokens(chunk.to(self.load_device)).to(mm.intermediate_device(), torch.float32)
                  for chunk in tokens.split(DECODE_CHUNK)]
        return torch.cat(images)


class DalleMiniLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = list(DALLE_REPOS) + [name for name in _local_dalle_models() if name not in DALLE_REPOS]
        return {
            "required": {
                "model": (models, {"default": "mini", "tooltip": "mini: 0.4B parameters, 1.75 GB download. "
                          "mega: 2.6B parameters, 10.4 GB download, better images. Downloaded from Hugging Face on "
                          "first use into ComfyUI/models/dalle_mini. Other folders placed there are listed too."}),
                "precision": (["auto", "fp32", "fp16", "bf16"], {"default": "auto", "tooltip": "auto uses fp16 on "
                              "GPUs that handle it well and fp32 elsewhere. The CPU always uses fp32."}),
            },
        }

    RETURN_TYPES = ("DALLE_MINI_MODEL", "DALLE_MINI_VQGAN")
    RETURN_NAMES = ("dalle_model", "vqgan")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Loads the DALL·E Mini / Mega text-to-image-token transformer and the VQGAN that turns tokens into pixels."

    def load(self, model, precision):
        dalle_folder = _model_folder(model, (CONFIG_NAME, WEIGHTS_NAME, TOKENIZER_NAME), DALLE_REPOS.get(model))
        vqgan_folder = _model_folder(VQGAN_NAME, (CONFIG_NAME, WEIGHTS_NAME), VQGAN_REPO)
        return (DalleModel(dalle_folder, precision), DalleVQGAN(vqgan_folder))


class DalleMiniTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dalle_model": ("DALLE_MINI_MODEL",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                    "default": "an armchair in the shape of an avocado",
                                    "tooltip": "Only the first 64 tokens are used. Leave empty for unconditional."}),
            },
        }

    RETURN_TYPES = ("DALLE_MINI_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Normalises and tokenizes a prompt, then runs it through the DALL·E Mini text encoder."

    def encode(self, dalle_model, text):
        return (dalle_model.encode(text),)


class DalleMiniSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dalle_model": ("DALLE_MINI_MODEL",),
                "positive": ("DALLE_MINI_CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                                       "tooltip": "Number of images. Each is sampled independently from the prompt."}),
                "condition_scale": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 0.5, "tooltip":
                                    "Super-conditioning (classifier-free guidance on the token logits). Higher follows "
                                    "the prompt more closely with less variety. 1.0 disables it and doubles the speed."}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 16384, "tooltip":
                          "Sample each token from the k most likely codes. 0 considers all of them."}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip":
                          "Nucleus sampling: keep the most likely codes whose probabilities add up to this. 1.0 is off."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip":
                                "Below 1 is more conservative, above 1 more varied. 0 always takes the most likely code."}),
            },
            "optional": {
                "negative": ("DALLE_MINI_CONDITIONING", {"tooltip":
                             "What to steer away from. Defaults to the empty prompt the model was trained with."}),
            },
        }

    RETURN_TYPES = ("DALLE_MINI_TOKENS",)
    RETURN_NAMES = ("tokens",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = "Autoregressively samples a 16x16 grid of VQGAN image tokens for each image."

    def sample(self, dalle_model, positive, seed, batch_size, condition_scale, top_k, top_p, temperature, negative=None):
        config = dalle_model.config
        guided = condition_scale != 1.0
        if guided and negative is None:
            negative = dalle_model.unconditional()

        rows = batch_size * (2 if guided else 1)
        cache_bytes = rows * config.decoder_layers * 2 * config.image_length * config.d_model * dalle_model.dtype.itemsize
        mm.load_models_gpu([dalle_model.patcher], memory_required=2 * cache_bytes + 64 * 1024 * 1024)

        cond, cond_mask = dalle_model.prepare(positive, batch_size)
        uncond, uncond_mask = dalle_model.prepare(negative, batch_size) if guided else (None, None)

        # Noise comes from a CPU generator, so it does not depend on the GPU in use. Row i depends only on
        # (seed, i): raising batch_size keeps the earlier images.
        uniform = torch.rand(batch_size, config.image_length, generator=torch.Generator().manual_seed(seed))

        progress = comfy.utils.ProgressBar(config.image_length)

        def callback(step, total):
            mm.throw_exception_if_processing_interrupted()
            progress.update_absolute(step, total)

        tokens = dalle_model.model.generate(
            cond, cond_mask, uncond, uncond_mask, uniform=uniform, condition_scale=condition_scale,
            top_k=top_k, top_p=top_p, temperature=temperature, callback=callback)
        return (tokens,)


class DalleMiniDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vqgan": ("DALLE_MINI_VQGAN",),
                "tokens": ("DALLE_MINI_TOKENS",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Decodes DALL·E Mini image tokens into 256x256 images with the VQGAN."

    def decode(self, vqgan, tokens):
        return (vqgan.decode(tokens),)


_clip_cache = {}


def _load_clip(name):
    if name not in _clip_cache:
        from transformers import AutoTokenizer, CLIPModel
        logging.info(f"[DALL-E Mini] loading {name}")
        model = CLIPModel.from_pretrained(name).eval().requires_grad_(False)
        _clip_cache[name] = (model, AutoTokenizer.from_pretrained(name))
    return _clip_cache[name]


class DalleMiniCLIPRank:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "",
                                    "tooltip": "The prompt the images are scored against."}),
                "clip_model": (CLIP_MODELS, {"default": CLIP_MODELS[0],
                               "tooltip": "Downloaded from Hugging Face on first use."}),
                "top_n": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Keep only the best n images. 0 keeps all."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "scores")
    FUNCTION = "rank"
    CATEGORY = CATEGORY
    DESCRIPTION = "Sorts images by how well CLIP thinks they match a prompt, best first."

    def rank(self, images, text, clip_model, top_n):
        model, tokenizer = _load_clip(clip_model)
        device = mm.get_torch_device()
        size = model.config.vision_config.image_size

        # CLIP preprocessing: shortest side to `size`, centre crop, normalise
        pixels = images.movedim(-1, 1).to(torch.float32)
        scale = size / min(pixels.shape[-2:])
        pixels = F.interpolate(pixels, size=(max(size, round(pixels.shape[-2] * scale)), max(size, round(pixels.shape[-1] * scale))),
                               mode="bicubic", antialias=True).clamp(0.0, 1.0)
        top, left = (pixels.shape[-2] - size) // 2, (pixels.shape[-1] - size) // 2
        pixels = pixels[:, :, top:top + size, left:left + size]
        pixels = (pixels - torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)) / torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        tokens = tokenizer([text], padding="max_length", max_length=77, truncation=True, return_tensors="pt")

        mm.free_memory(mm.module_size(model), device)
        model.to(device)
        try:
            scores = torch.cat([
                model(input_ids=tokens["input_ids"].to(device), attention_mask=tokens["attention_mask"].to(device),
                      pixel_values=chunk.to(device)).logits_per_image[:, 0].float().cpu()
                for chunk in pixels.split(16)])
        finally:
            model.to("cpu")

        order = scores.argsort(descending=True)
        if top_n > 0:
            order = order[:top_n]
        return (images[order], ", ".join(f"{scores[i]:.2f}" for i in order))


NODE_CLASS_MAPPINGS = {
    "DalleMiniLoader": DalleMiniLoader,
    "DalleMiniTextEncode": DalleMiniTextEncode,
    "DalleMiniSampler": DalleMiniSampler,
    "DalleMiniDecode": DalleMiniDecode,
    "DalleMiniCLIPRank": DalleMiniCLIPRank,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DalleMiniLoader": "DALL·E Mini Loader",
    "DalleMiniTextEncode": "DALL·E Mini Text Encode",
    "DalleMiniSampler": "DALL·E Mini Sampler",
    "DalleMiniDecode": "DALL·E Mini VQGAN Decode",
    "DalleMiniCLIPRank": "DALL·E Mini CLIP Rank",
}
