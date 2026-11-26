# ------------------------------------------------------------------------
# COMET3D: detrpose_v2 for RF-DETR
# Copyright (c) 2026 Ashesh Vasalya (avasalya, asheshvasalya@gmail.com). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Contract tests for ``keypoint_decoder="detrpose_v2"`` (box stream + original-DETRPose pose stream)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.models.postprocess import PostProcess
from rfdetr.models.transformer import Transformer
from rfdetr.training.callbacks.resolution_schedule import resolution_for_epoch
from rfdetr.training.module_model import FlatCosineLambda
from rfdetr.utilities.tensors import NestedTensor

D, B, NQ, K, L, G, M = 32, 2, 8, 5, 3, 2, 4
SHAPES = [(8, 8)]
POSE_SHAPES = [(16, 16), (8, 8), (4, 4)]


def _build(ddq: bool = False, ddq_mode: str = "box") -> LWDETR:
    """Tiny Transformer + LWDETR with a stub backbone returning one box level and three pose levels.

    Examples:
        >>> _build().pose_stream_enabled
        True
    """
    torch.manual_seed(0)
    feats = [NestedTensor(torch.randn(B, D, h, w), torch.zeros(B, h, w, dtype=torch.bool)) for h, w in SHAPES]
    pose_feats = [NestedTensor(torch.randn(B, D, h, w), torch.zeros(B, h, w, dtype=torch.bool)) for h, w in POSE_SHAPES]
    backbone = MagicMock(return_value=(feats, [torch.randn(B, D, h, w) for h, w in SHAPES], pose_feats))
    transformer = Transformer(
        d_model=D,
        sa_nhead=4,
        ca_nhead=4,
        num_queries=NQ,
        num_decoder_layers=L,
        dim_feedforward=64,
        return_intermediate_dec=True,
        group_detr=G,
        two_stage=True,
        num_feature_levels=1,
        dec_n_points=2,
        lite_refpoint_refine=True,
        bbox_reparam=True,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[K],
        keypoint_decoder="detrpose_v2",
        dual_projector_kp_only=True,
        detrpose_v2={
            "num_decoder_layers": L,
            "num_feature_levels": len(POSE_SHAPES),
            "nhead": 4,
            "dec_n_points": 2,
            "dim_feedforward": 64,
            "oks_sigmas": [0.05] * K,
            "spawn_topk": M,
            "spawn_topk_eval": M - 1,
            "ddq": ddq,
            "ddq_dense_topk": 2 * NQ,
            "ddq_mode": ddq_mode,
        },
    )
    return LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=2,
        num_queries=NQ,
        aux_loss=True,
        group_detr=G,
        two_stage=True,
        lite_refpoint_refine=True,
        bbox_reparam=True,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[K],
    )


def _targets() -> list[dict[str, torch.Tensor]]:
    """Two images: two persons with keypoints, and one person without keypoints (box-only source).

    Examples:
        >>> [len(t["labels"]) for t in _targets()]
        [2, 1]
    """
    torch.manual_seed(1)
    kps = torch.rand(2, K, 3)
    kps[..., 2] = 2.0
    box_only = torch.zeros(1, K, 3)
    box_only[..., 2] = -1.0
    return [
        {"labels": torch.tensor([0, 0]), "boxes": torch.tensor([[0.4, 0.4, 0.3, 0.5], [0.6, 0.5, 0.2, 0.4]]),
         "keypoints": kps},
        {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.6]]), "keypoints": box_only},
    ]


def _samples() -> NestedTensor:
    return NestedTensor(torch.randn(B, 3, 64, 64), torch.zeros(B, 64, 64, dtype=torch.bool))


class TestOutputs:
    def test_training_shapes(self) -> None:
        model = _build().train()
        out = model(_samples(), _targets())
        assert out["pred_keypoints"].shape == (B, G * NQ, K, 8)
        assert out["pred_logits"].shape == (B, G * NQ, 2)
        assert len(out["aux_outputs"]) == L - 1
        assert all(a["pred_keypoints"].shape == (B, G * NQ, K, 8) for a in out["aux_outputs"])
        assert out["pose_valid"].sum(dim=1).tolist() == [G * M] * B
        assert out["enc_outputs"]["pred_keypoints"].shape == (B, G * NQ, K, 8)
        assert "pose_pre_outputs" in out and "pose_dn" in out

    def test_eval_spawns_eval_topk_and_all_layers(self) -> None:
        model = _build().eval()
        with torch.no_grad():
            out = model(_samples())
        assert out["pred_keypoints"].shape == (B, NQ, K, 8)
        assert out["pose_valid"].sum(dim=1).tolist() == [M - 1] * B
        assert len(out["aux_outputs"]) == L - 1  # every layer visible to viz_decoder_layers.py
        assert "pose_dn" not in out

    def test_queries_without_pose_cannot_report_a_person(self) -> None:
        model = _build().eval()
        with torch.no_grad():
            out = model(_samples())
        person = out["pred_logits"][..., 0]
        assert (person[~out["pose_valid"]] < person[out["pose_valid"]].min()).all()

    def test_score_fusion_starts_at_box_stream_logit(self) -> None:
        model = _build().eval()
        assert model.pose_score_weight is not None and float(model.pose_score_weight) == 0.0

    def test_keypoints_inside_image_range_at_init(self) -> None:
        model = _build().eval()
        with torch.no_grad():
            out = model(_samples())
        xy = out["pred_keypoints"][out["pose_valid"]][..., :2]
        assert torch.isfinite(xy).all() and (xy > -0.5).all() and (xy < 1.5).all()

    def test_bf16_autocast_eval_outputs_float32_and_postprocesses(self) -> None:
        # Regression: validation under bf16 autocast crashed in PostProcess._apply_keypoint_trace_fusion
        # ("Index put requires the source and destination dtypes match ... BFloat16 ... Float").
        model = _build().eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(_samples())
            results = PostProcess(num_select=NQ, num_keypoints_per_class=[K])(out, torch.tensor([[64, 64]] * B))
        assert out["pred_keypoints"].dtype == torch.float32
        assert len(results) == B and results[0]["keypoints"].shape[-2] == K

    @pytest.mark.parametrize("mode", ["box", "pose"])
    def test_ddq_selection_keeps_query_count(self, mode: str) -> None:
        model = _build(ddq=True, ddq_mode=mode).train()
        out = model(_samples(), _targets())
        assert out["pred_logits"].shape == (B, G * NQ, 2)


def _criterion() -> SetCriterion:
    matcher = HungarianMatcher(
        cost_class=2, cost_bbox=5, cost_giou=2, num_keypoints_per_class=[K], keypoint_l1_loss_coef=1.0,
        cost_oks=4.0, keypoint_oks_sigmas=[0.05] * K,
    )
    weights = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2, "loss_keypoints_l1": 1, "loss_pose_oks": 4,
               "loss_pose_vfl": 2}
    return SetCriterion(
        2, matcher=matcher, weight_dict=weights, focal_alpha=0.25,
        losses=["labels", "boxes", "cardinality", "keypoints", "pose"], group_detr=G, ia_bce_loss=True,
        num_keypoints_per_class=[K], keypoint_oks_sigmas=[0.05] * K,
    )


class TestLosses:
    def test_all_pose_terms_present_and_finite(self) -> None:
        model = _build().train()
        criterion = _criterion().train()
        losses = criterion(model(_samples(), _targets()), _targets())
        for key in ("loss_pose_oks", "loss_pose_vfl", "loss_pose_oks_0", "loss_pose_oks_enc", "loss_pose_oks_pre",
                    "loss_keypoints_l1_pre", "loss_pose_dn_oks", "loss_pose_dn_vfl", "loss_pose_dn_l1"):
            assert key in losses, key
            assert torch.isfinite(losses[key]), key
        assert "loss_pose_vfl_enc" not in losses  # encoder layer has no pose score

    def test_pose_stream_receives_gradient(self) -> None:
        model = _build().train()
        criterion = _criterion().train()
        losses = criterion(model(_samples(), _targets()), _targets())
        total = sum(criterion.weight_dict.get(k, 1.0) * v for k, v in losses.items() if "dn" in k or "pose" in k)
        total.backward()
        stream = model.transformer.pose_stream
        assert stream.pose_embed[-1].layers[-1].weight.grad is not None
        assert stream.class_embed[-1].weight.grad is not None

    def test_oks_is_one_for_perfect_prediction(self) -> None:
        from rfdetr.models.heads.detrpose_v2 import keypoint_oks

        target = torch.rand(3, K, 3)
        target[..., 2] = 2.0
        oks = keypoint_oks(target[..., :2], target, torch.full((3,), 0.1), torch.full((K,), 0.05))
        torch.testing.assert_close(oks, torch.ones(3), atol=1e-5, rtol=0)


class TestCDN:
    def test_only_persons_with_keypoints_become_dn_queries(self) -> None:
        from rfdetr.models.heads.detrpose_v2 import prepare_for_cdn

        out = prepare_for_cdn(
            _targets(), 20, 0.5, num_queries=M, num_classes=1, num_keypoints=K, hidden_dim=D,
            label_enc=torch.nn.Embedding(2, D), pose_enc=torch.nn.Embedding(K, D), img_dim=(64, 64),
            sigmas=torch.full((K,), 0.05), person_label=0, device=torch.device("cpu"),
        )
        assert out is not None
        tokens, refs, mask, meta = out
        assert meta["known_targets"][1].numel() == 0  # box-only person
        assert tokens.shape == (B, meta["pad_size"], 1 + K, D)
        assert mask.shape == (meta["pad_size"] + M, meta["pad_size"] + M)
        assert mask[meta["pad_size"]:, : meta["pad_size"]].all()  # matching queries cannot see DN


class TestTrainingUtilities:
    def test_flat_cosine_shape_and_scale(self) -> None:
        f = FlatCosineLambda(warmup_steps=10, total_steps=100, flat_ratio=0.5, min_factor=0.1)
        assert f(0) == 0.0 and f(10) == 1.0 and f(49) == 1.0 and f(75) < 1.0
        f.scale = 0.3
        assert f(30) == pytest.approx(0.3)

    def test_resolution_schedule(self) -> None:
        assert [resolution_for_epoch({10: 672}, e, 576) for e in (0, 9, 10, 40)] == [576, 576, 672, 672]
