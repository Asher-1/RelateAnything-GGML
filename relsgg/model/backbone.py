"""DINOv3 backbone and the relation interaction block."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.v2 import Normalize

from.geometry import ScenePosEnc

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
# Hidden states fused into the scene map, relative to the last block.
LAYER_OFFSETS = [-6, -3, -1]
# ConvNeXt: all four stage maps (hidden_states[0] is the input image).
CONVNEXT_STAGE_OFFSETS = [-4, -3, -2, -1]


def _enable_block_checkpointing(block_cls: type) -> None:
    """Patch ``block_cls.forward`` (once) so an instance with
    ``_ra_grad_ckpt`` set runs under non-reentrant activation checkpointing
    while training. Instances without the flag, and eval passes, are untouched."""
    if getattr(block_cls, "_ra_plain_forward", None) is not None:
        return
    plain = block_cls.forward

    def forward(self, *args, **kwargs):
        if getattr(self, "_ra_grad_ckpt", False) and self.training and torch.is_grad_enabled():
            return checkpoint(plain, self, *args, use_reentrant=False, **kwargs)
        return plain(self, *args, **kwargs)

    block_cls._ra_plain_forward = plain
    block_cls.forward = forward


class Backbone(nn.Module):
    """DINOv3 tower, fully fine-tuned, read at several depths.

    ViT: three hidden states, LayerNorm-ed and combined with a softmax over
    three learned scalars, so their share of the scene map does not depend on
    the growth of activation norms with depth.

    ConvNeXt (``model_type == "dinov3_convnext"``): the four stage maps, at
    strides 4/8/16/32, are brought to the stride-``patch_size`` grid —
    space-to-depth for the finer stages, bilinear for the coarser one — then
    each goes through its own 1x1 projection to the deepest stage's width and
    an affine LayerNorm before the same softmax combiner. Without the norm the
    stages differ 13x in scale, the stride-4/8 maps end up with 9 % of the sum,
    and the combiner never leaves its uniform init. The projections, their
    norms and the combiner are trained from scratch; see ``from_scratch``.

    Args:
        model_name: Hugging Face id or local directory of the DINOv3 tower.
        pretrained: load the pretrained weights (training); ``False`` builds
                    the architecture from ``config.json`` only, for released
                    checkpoints that carry every weight themselves.
    """

    def __init__(self, model_name: str, patch_size: int = 16, pretrained: bool = True,
                 layer_offsets: List[int] = LAYER_OFFSETS):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.patch_size = patch_size
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        cfg = self.model.config
        self.is_convnext = cfg.model_type == "dinov3_convnext"
        self.layer_offsets = list(CONVNEXT_STAGE_OFFSETS if self.is_convnext else layer_offsets)
        # The taps read hidden states, so the final norm and the mask token
        # never receive a gradient; freeze them so the audit does not report
        # them as dead.
        for attr in ("norm", "layernorm", "layer_norm"):
            mod = getattr(self.model, attr, None)
            if mod is not None:
                mod.requires_grad_(False)
        emb = getattr(self.model, "embeddings", None)
        mask_token = getattr(emb, "mask_token", None) if emb is not None else None
        if mask_token is not None:
            mask_token.requires_grad_(False)

        self.d_model: int = cfg.hidden_sizes[-1] if self.is_convnext else cfg.hidden_size
        self._normalize = Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        self.layer_weights = nn.Parameter(torch.zeros(len(self.layer_offsets)))
        if self.is_convnext:
            # Stage i has stride 4 * 2**i, so its ratio to the patch grid is
            # fixed: 4, 2, 1, 0.5 at patch size 16. A stage r > 1 times finer
            # is folded r x r into channels, which keeps its spatial detail
            # (an area average keeps 55 % of the stride-4 map's variance).
            self.stage_ratios = [patch_size / (4 * 2 ** i) for i in range(len(cfg.hidden_sizes))]
            in_ch = [int(c * r * r) if r > 1 else c
                     for c, r in zip(cfg.hidden_sizes, self.stage_ratios)]
            self.stage_proj = nn.ModuleList([nn.Conv2d(c, self.d_model, 1) for c in in_ch])
            self.stage_norm = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in in_ch])

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        """Recompute every ConvNeXt block in the backward pass instead of
        storing its activations: the same gradients for about one extra forward
        of the tower. ConvNeXt-B does not otherwise fit 32 images at the 672 px
        multi-scale rung on a 48 GB card, and the per-GPU batch is part of the
        recipe (the contrastive set is batch-local).

        A flag on each block rather than a wrapper module or a per-instance
        ``forward``: the parameter names stay those of the pretrained tower, and
        the EMA's deep copy keeps calling its own weights."""
        if not self.is_convnext:
            raise NotImplementedError("activation checkpointing is implemented for ConvNeXt only")
        stages = [m for m in self.model.modules()
                  if hasattr(m, "downsample_layers") and hasattr(m, "layers")]
        assert len(stages) == len(self.model.config.hidden_sizes), "ConvNeXt stages not found"
        for stage in stages:
            for layer in stage.layers:
                _enable_block_checkpointing(type(layer))
                layer._ra_grad_ckpt = enabled

    def from_scratch(self, name: str) -> bool:
        """Whether parameter ``name`` (relative to this module) is trained from
        scratch rather than adapted from the pretrained tower. Such parameters
        belong at the head's learning rate: at the backbone's they starve. The
        ViT combiner stays with the backbone, where every released ViT run
        trained it."""
        return self.is_convnext and not name.startswith("model.")

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Float images in [0, 1] -> ImageNet-normalised."""
        return self._normalize(images)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Normalised ``[B, 3, H, W]`` -> fused patch features ``[B, H/p, W/p, d]``."""
        B, _, H, W = images.shape
        h, w = H // self.patch_size, W // self.patch_size
        hidden_states = self.model(pixel_values=images, output_hidden_states=True).hidden_states
        total = len(hidden_states)
        indices = [total + off if off < 0 else off for off in self.layer_offsets]
        weights = F.softmax(self.layer_weights, dim=0)
        if self.is_convnext:
            return self._fuse_stages([hidden_states[i] for i in indices], weights, h, w)
        n_patch = h * w
        # Patch tokens are the last h*w tokens (CLS and register tokens first).
        taps = [F.layer_norm(hidden_states[i][:, -n_patch:,:], (self.d_model,))
                for i in indices]
        fused = sum(weights[i] * t for i, t in enumerate(taps))
        return fused.reshape(B, h, w, self.d_model)

    def _fuse_stages(self, stages: List[torch.Tensor], weights: torch.Tensor,
                     h: int, w: int) -> torch.Tensor:
        """ConvNeXt stage maps ``[B, C_i, h_i, w_i]`` -> ``[B, h, w, d]``.

        With H and W multiples of the patch size the finer stages are exactly
        ``r`` times the grid; the stride-32 stage is half of it, rounded down
        when the grid is odd (multi-scale rungs), which the resize absorbs."""
        fused = 0
        for i, f in enumerate(stages):
            r = self.stage_ratios[i]
            if r > 1:
                f = F.pixel_unshuffle(f, int(r))
            elif f.shape[-2:] != (h, w):
                f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
            f = self.stage_norm[i](self.stage_proj[i](f).permute(0, 2, 3, 1))
            fused = fused + weights[i] * f
        return fused


class RelationInteractionBlock(nn.Module):
    """Refinement of the pair queries after the relation transformer.

    Stage 1 (``n_dep`` self-attention layers): pairs attend to each other, so
    co-occurring relations can support or suppress one another.
    Stage 2 (``n_gnd`` joint layers): the queries and the scene tokens form one
    sequence passed through a self-attention layer, and the query positions
    are read back; queries and scene update each other.
    """

    def __init__(self, d_model: int, scene_dim: Optional[int] = None, n_dep: int = 2,
                 n_gnd: int = 1, n_heads: int = 8, ffn_ratio: float = 2.0,
                 dropout: float = 0.2) -> None:
        super().__init__()
        ffn_dim = int(d_model * ffn_ratio)
        if scene_dim is not None and scene_dim != d_model:
            self.scene_proj: nn.Module = nn.Linear(scene_dim, d_model, bias=False)
            nn.init.xavier_uniform_(self.scene_proj.weight)
        else:
            self.scene_proj = nn.Identity()
        self.scene_pe = ScenePosEnc(d_model)

        def _layer():
            return nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)

        self.dep_layers = nn.ModuleList([_layer() for _ in range(n_dep)])
        self.gnd_layers = nn.ModuleList([_layer() for _ in range(n_gnd)])

    def forward(self, queries: torch.Tensor, image_tokens: torch.Tensor,
                query_padding_mask: Optional[torch.Tensor] = None,
                grid_hw: Optional[tuple] = None) -> torch.Tensor:
        x = queries
        mem = self.scene_proj(image_tokens)
        h, w = grid_hw
        mem = mem + self.scene_pe(h, w, mem.device, mem.dtype)
        for layer in self.dep_layers:
            x = layer(x, src_key_padding_mask=query_padding_mask)
        n_q = x.shape[1]
        for layer in self.gnd_layers:
            joint = torch.cat([x, mem], dim=1)
            joint_mask = None
            if query_padding_mask is not None:
                scene_no_pad = torch.zeros(x.shape[0], mem.shape[1], dtype=torch.bool,
                                           device=x.device)
                joint_mask = torch.cat([query_padding_mask, scene_no_pad], dim=1)
            x = layer(joint, src_key_padding_mask=joint_mask)[:,:n_q]
        return x
