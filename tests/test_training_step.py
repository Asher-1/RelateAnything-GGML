"""One training step of the real objective, on a tiny model.

Training a released model needs GPUs, so what CI can check is that the
objective runs end to end and that every trainable parameter receives a
gradient from it. The second half is the point: a module the loss never
reaches trains as dead weight and nothing else notices, because rank-based
metrics are invariant to whatever it does.
"""
import json
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from relsgg.config import RelSGGConfig          # noqa: E402
from relsgg.model import RelSGG                 # noqa: E402
from relsgg.training.losses import PredicateOntology  # noqa: E402

V = 12          # predicates
N = 5           # boxes per image
B = 2           # images


def _tiny_backbone_dir(tmp, family="vit"):
    """A DINOv3 config small enough to run on a CPU, built from config only."""
    if family == "convnext":
        cfg = {"model_type": "dinov3_convnext", "hidden_sizes": [8, 16, 24, 32],
               "depths": [1, 1, 1, 1], "num_channels": 3, "drop_path_rate": 0.0}
        with open(os.path.join(tmp, "config.json"), "w") as fh:
            json.dump(cfg, fh)
        return tmp
    cfg = {"model_type": "dinov3_vit", "hidden_size": 32, "intermediate_size": 64,
           "num_hidden_layers": 2, "num_attention_heads": 2, "patch_size": 16,
           "image_size": 64, "num_channels": 3, "hidden_act": "silu",
           "use_gated_mlp": True, "num_register_tokens": 1, "layer_norm_eps": 1e-5,
           "rope_theta": 100.0, "layerscale_value": 1.0, "query_bias": True,
           "key_bias": False, "value_bias": True, "proj_bias": True, "mlp_bias": True,
           "pos_embed_rescale": 2.0, "drop_path_rate": 0.0}
    with open(os.path.join(tmp, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    return tmp


def _ontology():
    """Identity positives, no down-weighting, no inverses — the shape the loss
    reads, with every estimated table at its neutral value."""
    pos_w = torch.eye(V, dtype=torch.float16)
    neg_lw = torch.zeros(V, V, dtype=torch.float16)
    sym = torch.zeros(V)
    inverse = torch.zeros(V, V, dtype=torch.bool)
    return PredicateOntology([f"p{i}" for i in range(V)], pos_w, neg_lw, sym, inverse)


def _batch(device="cpu"):
    torch.manual_seed(0)
    images = torch.rand(B, 3, 64, 64)
    cx, cy = torch.rand(B, N) * 0.6 + 0.2, torch.rand(B, N) * 0.6 + 0.2
    boxes = torch.stack([cx, cy, torch.full((B, N), 0.2), torch.full((B, N), 0.2)], -1)
    box_counts = torch.full((B,), N, dtype=torch.long)
    targets = [{"relations": torch.tensor([[0, 1, 3], [1, 2, 7], [3, 4, 3]]),
                "rel_weights": torch.ones(3),
                "entity_labels": torch.arange(N),
                "src": 0} for _ in range(B)]
    return images, boxes, box_counts, targets


@pytest.fixture(scope="module", params=["vit", "convnext"])
def model(request):
    tmp = tempfile.mkdtemp(prefix="ra_test_backbone_")
    cfg = RelSGGConfig(backbone_model=_tiny_backbone_dir(tmp, request.param),
                       backbone_pretrained=False,
                       d_model=32, text_dim=16, geo_budget=12, final_budget=6,
                       n_self_layers=1, n_cross_layers=1, n_dep_layers=1, n_gnd_layers=1,
                       n_heads=2, deformable_points=2, deformable_heads=2,
                       deformable_nulls=1, proj_layers=2, dropout=0.0)
    m = RelSGG(cfg)
    m.vocab_head.set_vocabulary_matrix([f"p{i}" for i in range(V)], torch.randn(V, 16))
    m.install_losses(_ontology(), n_neg=8)
    m.set_object_vocabulary([f"c{i}" for i in range(N)], torch.randn(N, 16))
    return m


def test_one_step_produces_a_finite_loss(model):
    model.train()
    out = model(*_batch())
    assert torch.isfinite(out["loss"]), out["loss_dict"]
    for k, v in out["loss_dict"].items():
        assert torch.isfinite(v), f"{k} is not finite"
    assert out["logits"].shape == (B, model.config.final_budget, V)


def test_every_trainable_parameter_receives_gradient(model):
    """The default-on audit in the training loop, as a test."""
    model.train()
    model.zero_grad(set_to_none=True)
    model(*_batch())["loss"].backward()
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    # The pooling shape gate is inactive by design when no masks are given.
    dead = [n for n in dead if n != "spatial_pool.cov_lambda"]
    assert not dead, f"no gradient reached: {dead}"


def test_inference_needs_no_targets_and_no_labels(model):
    model.eval()
    images, boxes, box_counts, _ = _batch()
    with torch.no_grad():
        out = model(images, boxes, box_counts=box_counts, targets=None)
    assert "loss" not in out
    assert out["pair_logits"].shape == (B, model.config.final_budget)


def test_masks_change_the_geometry_features_only_where_they_differ(model):
    """A region raster of the box itself has to reproduce the box result: the
    geometry features take their box values when a region fills its box."""
    model.eval()
    images, boxes, box_counts, _ = _batch()
    g = 32
    cx, cy, w, h = boxes.unbind(-1)
    e = torch.arange(g + 1, dtype=torch.float32) / g
    ix = (torch.minimum((cx + w / 2).unsqueeze(-1), e[1:])
          - torch.maximum((cx - w / 2).unsqueeze(-1), e[:-1])).clamp(min=0)
    iy = (torch.minimum((cy + h / 2).unsqueeze(-1), e[1:])
          - torch.maximum((cy - h / 2).unsqueeze(-1), e[:-1])).clamp(min=0)
    cov = (iy.unsqueeze(-1) * ix.unsqueeze(-2)) * (g * g)
    with torch.no_grad():
        plain = model(images, boxes, box_counts=box_counts, targets=None)
        boxed = model(images, boxes, box_counts=box_counts, targets=None,
                      cov=cov, fill=torch.ones(B, N))
    assert torch.allclose(plain["logits"], boxed["logits"], atol=2e-4)


@pytest.mark.parametrize("size", [64, 80, 112])
def test_scene_map_is_on_the_patch_grid_at_every_rung(model, size):
    """Multi-scale rungs are multiples of 16, not of 32: an odd grid leaves
    the ConvNeXt stride-32 stage half a cell short, which the fusion absorbs."""
    model.eval()
    with torch.no_grad():
        f = model.backbone.extract(torch.rand(1, 3, size, size))
    assert f.shape == (1, size // 16, size // 16, model.backbone.d_model)


def test_only_the_pretrained_tower_gets_the_backbone_learning_rate(model):
    """A from-scratch module at the backbone's learning rate starves; the
    first ConvNeXt run trained its stage fusion that way."""
    from types import SimpleNamespace
    from relsgg.training.setup import build_optimizer
    args = SimpleNamespace(lr=4e-4, backbone_lr=1e-5, weight_decay=1e-4)
    opt = build_optimizer(model, args)
    lr_of = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        tower = name.startswith("backbone.model.") or (
            name == "backbone.layer_weights" and not model.backbone.is_convnext)
        assert lr_of[id(p)] == (1e-5 if tower else 4e-4), name


def test_convnext_checkpointing_keeps_the_gradients_and_the_ema_copy(model, monkeypatch):
    """Activation checkpointing must change memory only: the same backbone
    gradients, recomputation actually taken while training, and a deep copy
    (the EMA) that still runs on its own weights."""
    import copy
    import relsgg.model.backbone as bbmod
    bb = model.backbone
    if not bb.is_convnext:
        with pytest.raises(NotImplementedError):
            bb.set_grad_checkpointing(True)
        return
    x = torch.rand(2, 3, 64, 64)

    def grads():
        bb.train()
        bb.zero_grad(set_to_none=True)
        bb.extract(x).square().sum().backward()
        return {n: p.grad.clone() for n, p in bb.named_parameters() if p.grad is not None}

    plain = grads()
    calls = []
    real = bbmod.checkpoint
    monkeypatch.setattr(bbmod, "checkpoint", lambda *a, **k: calls.append(1) or real(*a, **k))
    bb.set_grad_checkpointing(True)
    try:
        ckpt = grads()
        n_blocks = sum(len(s.layers) for s in bb.model.modules() if hasattr(s, "downsample_layers"))
        assert len(calls) == n_blocks, "checkpointing was not used for every block"
        assert plain.keys() == ckpt.keys()
        for n in plain:
            assert torch.allclose(plain[n], ckpt[n], atol=1e-6), n
        # The EMA is a deep copy: perturbing it must change ITS output, which
        # it would not if the copied blocks still called the originals.
        twin = copy.deepcopy(bb).eval()
        with torch.no_grad():
            ref = twin.extract(x)
            for p in twin.model.parameters():
                p.add_(0.1)
            assert not torch.allclose(ref, twin.extract(x))
            calls.clear()
            bb.eval()
            bb.extract(x)
        assert not calls, "checkpointing ran outside training"
    finally:
        bb.set_grad_checkpointing(False)
