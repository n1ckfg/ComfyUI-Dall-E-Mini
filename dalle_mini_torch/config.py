"""Model configurations, read from the ``config.json`` shipped with each checkpoint."""

import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Optional

LN_TYPES = ("layernorm", "rmsnorm")
LN_POSITIONS = ("normformer", "swinv2", "cogview", "postln", "preln", "subln")


def _from_json(cls, path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class DalleBartConfig:
    # Defaults mirror dalle_mini.model.configuration.DalleBartConfig: checkpoints may omit
    # any key that was left at its default.
    normalize_text: bool = False
    encoder_vocab_size: int = 50264
    image_vocab_size: int = 16384
    image_length: int = 256
    max_text_length: int = 64
    encoder_layers: int = 12
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_layers: int = 12
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    activation_function: str = "gelu"
    d_model: int = 1024
    scale_embedding: bool = False
    tie_word_embeddings: bool = False
    use_bias: bool = False
    ln_type: str = "layernorm"
    ln_positions: str = "normformer"
    use_head_scale: bool = False
    use_cosine_attention: bool = False
    tau_init: float = 0.05
    use_absolute_position_embeddings: bool = True
    use_swin_position_embeddings: bool = False
    use_deepnet_scaling: bool = False
    use_glu: bool = True
    sinkhorn_iters: int = 1
    use_final_ln_encoder: bool = True
    use_final_ln_decoder: bool = True
    force_ln_scale: bool = False
    use_scan: Optional[bool] = None
    decoder_start_token_id: Optional[int] = None

    def __post_init__(self):
        if self.ln_positions == "deepnet":
            self.ln_positions = "postln"
        if self.ln_type not in LN_TYPES:
            raise ValueError(f"ln_type must be one of {LN_TYPES}, got {self.ln_type!r}")
        if self.ln_positions not in LN_POSITIONS:
            raise ValueError(f"ln_positions must be one of {LN_POSITIONS}, got {self.ln_positions!r}")
        if self.tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported: text and image vocabularies differ")
        if self.use_scan is None:
            self.use_scan = self.ln_positions != "swinv2"
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.image_vocab_size

    @property
    def num_image_codes(self):
        """Number of real VQGAN codes. ``image_vocab_size`` can be padded past the codebook, and the
        decoder start token is the first id after the real codes."""
        return min(self.image_vocab_size, self.decoder_start_token_id)

    @classmethod
    def from_json(cls, path):
        return _from_json(cls, path)


@dataclass
class VQGANConfig:
    ch: int = 128
    out_ch: int = 3
    in_channels: int = 3
    num_res_blocks: int = 2
    resolution: int = 256
    z_channels: int = 256
    ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    n_embed: int = 1024
    embed_dim: int = 256
    resamp_with_conv: bool = True

    @property
    def num_resolutions(self):
        return len(self.ch_mult)

    @classmethod
    def from_json(cls, path):
        return _from_json(cls, path)
