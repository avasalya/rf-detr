# ------------------------------------------------------------------------
# COMET3D: detrpose_v2 for RF-DETR
# Copyright (c) 2026 Ashesh Vasalya (avasalya, asheshvasalya@gmail.com). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Numerical parity of the vendored DETRPose components with the original code in ``external/DETRPose``.

Each test builds the original module and ours, copies one ``state_dict`` into both and compares the outputs on the
same random inputs (eval mode: the original's in-place positional add is only autograd-safe under AMP). Skipped when
the DETRPose checkout (COMET3D ``external/DETRPose``) is not next to this repository.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from rfdetr.models.heads import detrpose_v2 as ours

DETRPOSE_SRC = Path(__file__).resolve().parents[3] / "DETRPose"


def _original() -> ModuleType:
    """Import the original ``src.models.detrpose.transformer`` (skips the test when unavailable).

    Examples:
        >>> callable(_original)  # doctest: +SKIP
        True
    """
    if not (DETRPOSE_SRC / "src" / "models" / "detrpose" / "transformer.py").is_file():
        pytest.skip("external/DETRPose not found")
    if str(DETRPOSE_SRC) not in sys.path:
        sys.path.insert(0, str(DETRPOSE_SRC))
    # Register the parent packages as bare namespaces so their __init__ (datasets, xtcocotools, …) never runs; only
    # the decoder modules and their relative imports are loaded.
    for name, path in (
        ("src", "src"),
        ("src.misc", "src/misc"),
        ("src.models", "src/models"),
        ("src.models.detrpose", "src/models/detrpose"),
    ):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = [str(DETRPOSE_SRC / path)]
            sys.modules[name] = package
    try:
        return importlib.import_module("src.models.detrpose.transformer")
    except ImportError as exc:  # missing optional dependency of the original package
        pytest.skip(f"cannot import DETRPose: {exc}")


D, H, P, LEVELS, K = 32, 4, 4, 2, 5
SHAPES = [[6, 6], [3, 3]]


def _memory(bs: int) -> list[torch.Tensor]:
    """Per-level values ``[bs * heads, head_dim, H * W]`` as the original transformer builds them.

    Examples:
        >>> [v.shape for v in _memory(1)]
        [torch.Size([4, 8, 36]), torch.Size([4, 8, 9])]
    """
    memory = torch.randn(bs, sum(h * w for h, w in SHAPES), D)
    value = memory.unflatten(2, (H, -1)).permute(0, 2, 3, 1).flatten(0, 1)
    return list(value.split([h * w for h, w in SHAPES], dim=-1))


class TestComponents:
    def test_gate(self) -> None:
        orig = _original()
        a, b = orig.Gate(D), ours.Gate(D)
        b.load_state_dict(a.state_dict())
        with torch.no_grad():
            a.gate.weight.normal_()
        b.load_state_dict(a.state_dict())
        x1, x2 = torch.randn(2, 3, D), torch.randn(2, 3, D)
        torch.testing.assert_close(b(x1, x2), a(x1, x2))

    def test_integral_and_distance2pose(self) -> None:
        orig = _original()
        up, scale = torch.tensor([0.5]), torch.tensor([4.0])
        project = ours.weighting_function(32, up, scale)
        torch.testing.assert_close(project, orig.weighting_function(32, up, scale))
        corners = torch.randn(2, 4, K, 2 * 33)  # B * nq even, as the original's reshape(-1, 4) requires
        ref = torch.rand(2, 4, K, 2)
        expected = orig.distance2pose(ref, orig.Integral(32)(corners, project), scale)
        torch.testing.assert_close(ours.distance2pose(ref, ours.Integral(32)(corners, project), scale), expected)

    def test_lqe(self) -> None:
        orig = _original()
        a, b = orig.LQE(4, 64, 2, K), ours.LQE(4, 64, 2, K)
        with torch.no_grad():
            for p in a.parameters():
                p.normal_()
        b.load_state_dict(a.state_dict())
        scores, poses, feat = torch.randn(2, 3, 1), torch.rand(2, 3, K, 2), torch.randn(2, D, 8, 8)
        torch.testing.assert_close(b(scores, poses, feat), a(scores, poses, feat))


def _random_layer_params(layer: torch.nn.Module) -> None:
    """Replace the zero/xavier init by random values so every branch contributes.

    Examples:
        >>> m = torch.nn.Linear(2, 2); _random_layer_params(m); bool(m.weight.abs().sum() > 0)
        True
    """
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p) * 0.1)


class TestDecoder:
    def test_layer(self) -> None:
        orig = _original()
        torch.manual_seed(0)
        a = orig.DeformableTransformerDecoderLayer(D, 64, 0.0, "relu", LEVELS, H, P).eval()
        b = ours.DeformableTransformerDecoderLayer(D, 64, 0.0, "relu", LEVELS, H, P).eval()
        _random_layer_params(a)
        b.load_state_dict(a.state_dict())
        bs, nq = 2, 3
        tgt = torch.randn(bs, nq, 1 + K, D)
        pos = torch.randn(bs, nq, K, D)
        ref = torch.rand(bs, nq, 1, 1 + K, 2)
        memory = _memory(bs)
        with torch.no_grad():
            expected = a(tgt.clone(), pos, ref, memory=memory, memory_spatial_shapes=SHAPES)
            actual = b(tgt.clone(), pos, ref, memory=memory, memory_spatial_shapes=SHAPES)
        torch.testing.assert_close(actual, expected)

    def test_full_decoder_loop(self) -> None:
        """6-layer loop incl. FDR refinement, LQE scores and the in-place positional side effect."""
        orig = _original()
        torch.manual_seed(0)
        layers, bs, nq = 3, 2, 4
        a_layer = orig.DeformableTransformerDecoderLayer(D, 64, 0.0, "relu", LEVELS, H, P)
        a = orig.TransformerDecoder(a_layer, layers, hidden_dim=D, num_body_points=K).eval()
        b = ours.PoseTransformerDecoder(
            ours.DeformableTransformerDecoderLayer(D, 64, 0.0, "relu", LEVELS, H, P), layers, D, K
        ).eval()
        _random_layer_params(a)
        b.load_state_dict(a.state_dict())
        heads = {
            "pre_pose_head": orig.MLP(D, D, 2, 3),
            "pose_head": torch.nn.ModuleList([orig.MLP(D, D, 2 * 33, 3) for _ in range(layers)]),
            "class_head": torch.nn.ModuleList([torch.nn.Linear(D, 2) for _ in range(layers)]),
            "lqe_head": torch.nn.ModuleList([orig.LQE(4, 64, 2, K) for _ in range(layers)]),
        }
        for head in heads.values():
            _random_layer_params(head)
        up, scale = torch.tensor([0.5]), torch.tensor([4.0])
        project = orig.weighting_function(32, up, scale)
        tgt = torch.randn(bs, nq, 1 + K, D)
        refs = torch.rand(bs, nq, 1 + K, 2) * 0.8 + 0.1
        memory = _memory(bs)
        feat = torch.randn(bs, D, 6, 6)
        with torch.no_grad():
            poses, logits, *_ = a(
                tgt.clone(), memory, refs, heads["pre_pose_head"], heads["pose_head"], heads["class_head"],
                heads["lqe_head"], feat, orig.Integral(32), up, scale, 32, project, spatial_shapes=SHAPES,
            )
            out = b(
                tgt.clone(), memory, refs, heads["pre_pose_head"], heads["pose_head"], heads["class_head"],
                heads["lqe_head"], feat, ours.Integral(32), scale, project, spatial_shapes=SHAPES,
            )
        torch.testing.assert_close(out["poses"][-1], poses[-1])
        torch.testing.assert_close(out["logits"][-1], logits[-1])
