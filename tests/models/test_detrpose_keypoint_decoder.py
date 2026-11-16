# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Contract tests for the DETRPose-style keypoint decoder (``keypoint_decoder="detrpose"``)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.transformer import Transformer
from rfdetr.utilities.tensors import NestedTensor

D, B, NQ, K, L, G = 32, 2, 6, 17, 3, 2


def _build(mode: str, registers: int = 0, across_topk: int = 100, two_stage: bool = True) -> LWDETR:
    """Build a tiny real Transformer + LWDETR (stub backbone, no padding so valid ratios are 1).

    Examples:
        >>> _build("detrpose").keypoint_decoder
        'detrpose'
    """
    torch.manual_seed(0)
    shapes = [(8, 8), (4, 4)]
    feats = [NestedTensor(torch.randn(B, D, h, w), torch.zeros(B, h, w, dtype=torch.bool)) for h, w in shapes]
    backbone = MagicMock(return_value=(feats, [torch.randn(B, D, h, w) for h, w in shapes], None))
    transformer = Transformer(
        d_model=D,
        sa_nhead=4,
        ca_nhead=4,
        num_queries=NQ,
        num_decoder_layers=L,
        dim_feedforward=64,
        return_intermediate_dec=True,
        group_detr=G,
        two_stage=two_stage,
        num_feature_levels=2,
        dec_n_points=2,
        lite_refpoint_refine=True,
        bbox_reparam=True,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[K],
        inter_instance_kp_attn=True,
        keypoint_decoder=mode,
        detrpose_across_topk=across_topk,
        num_registers=registers,
    )
    return LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=2,
        num_queries=NQ,
        aux_loss=True,
        group_detr=G,
        two_stage=two_stage,
        lite_refpoint_refine=True,
        bbox_reparam=True,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[K],
    )


def _capture_layer_kwargs(model: LWDETR) -> list[dict]:
    """Record the keyword arguments every decoder layer is called with.

    Examples:
        >>> len(_capture_layer_kwargs(_build("detrpose")))
        0
    """
    calls: list[dict] = []
    for layer in model.transformer.decoder.layers:
        layer.register_forward_pre_hook(lambda _m, _a, kw: calls.append(dict(kw)), with_kwargs=True)
    return calls


def _image() -> torch.Tensor:
    """Dummy image batch (the stub backbone ignores its content).

    Examples:
        >>> _image().shape
        torch.Size([2, 3, 32, 32])
    """
    return torch.ones(B, 3, 32, 32)


class TestStateDictContract:
    def test_new_parameters_are_exactly_gate_and_reference_head(self) -> None:
        gp, dp = set(_build("grouppose").state_dict()), set(_build("detrpose").state_dict())
        assert all(".kp_cross_attn_norm." in k for k in gp - dp), gp - dp  # replaced by kp_gate
        added = dp - gp
        assert added
        assert all(k.startswith("transformer.decoder.kp_ref_point_head.") or ".kp_gate." in k for k in added), added

    def test_shared_keypoint_head_not_registered_in_decoder(self) -> None:
        model = _build("detrpose")
        assert model.transformer.decoder._keypoint_head is model.keypoint_embed
        assert not any("_keypoint_head" in k for k in model.state_dict())

    def test_grouppose_checkpoint_warm_starts_detrpose(self) -> None:
        missing, unexpected = _build("detrpose").load_state_dict(_build("grouppose").state_dict(), strict=False)
        assert all(".kp_cross_attn_norm." in k for k in unexpected), unexpected
        assert all("kp_ref_point_head" in k or "kp_gate" in k for k in missing)


class TestOutputContract:
    @pytest.mark.parametrize("mode", ["grouppose", "detrpose"])
    @pytest.mark.parametrize("registers", [0, 2])
    def test_train_and_eval_shapes(self, mode: str, registers: int) -> None:
        model = _build(mode, registers=registers).train()
        out = model(_image())
        assert out["pred_keypoints"].shape == (B, NQ * G, K, 8)
        assert [a["pred_keypoints"].shape for a in out["aux_outputs"]] == [(B, NQ * G, K, 8)] * (L - 1)
        model.eval()
        with torch.no_grad():
            out = model(_image())
        assert out["pred_keypoints"].shape == (B, NQ, K, 8)
        assert torch.isfinite(out["pred_keypoints"]).all()

    def test_all_parameters_receive_gradients(self) -> None:
        model = _build("detrpose").train()
        out = model(_image())
        loss = out["pred_keypoints"].sum() + out["pred_boxes"].sum() + out["pred_logits"].sum()
        loss = loss + sum(a["pred_keypoints"].sum() for a in out["aux_outputs"])
        loss.backward()
        no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        # Encoder-side heads only get gradients from enc_outputs losses, not from this decoder-only loss.
        assert all(".enc_" in n or "_enc." in n for n in no_grad), no_grad


class TestReferenceChain:
    def test_layer0_references_are_encoder_keypoints_and_pose_centre(self) -> None:
        model = _build("detrpose").eval()
        calls = _capture_layer_kwargs(model)
        with torch.no_grad():
            out = model(_image())
        ref = calls[0]["kp_reference_points"].view(B, NQ, 1 + K, -1, 4)[:, :, :, 0]  # level 0
        enc_xy = out["enc_outputs"]["pred_keypoints"][..., :2]
        torch.testing.assert_close(ref[:, :, 1:, :2], enc_xy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ref[:, :, 0, :2], enc_xy.mean(dim=2), atol=1e-5, rtol=1e-5)

    def test_reference_spread_is_box_wh(self) -> None:
        model = _build("detrpose").eval()
        calls = _capture_layer_kwargs(model)
        with torch.no_grad():
            model(_image())
        ref = calls[0]["kp_reference_points"].view(B, NQ, 1 + K, -1, 4)[:, :, :, 0]
        box_wh = calls[0]["reference_points"][:, :, 0, 2:]
        torch.testing.assert_close(ref[..., 2:], box_wh[:, :, None].expand_as(ref[..., 2:]))

    def test_layer_prediction_becomes_next_layer_reference(self) -> None:
        model = _build("detrpose").eval()
        calls = _capture_layer_kwargs(model)
        with torch.no_grad():
            out = model(_image())
        preds = [a["pred_keypoints"] for a in out["aux_outputs"]] + [out["pred_keypoints"]]
        for layer_id in range(1, L):
            ref = calls[layer_id]["kp_reference_points"].view(B, NQ, 1 + K, -1, 4)[:, :, 1:, 0, :2]
            torch.testing.assert_close(ref, preds[layer_id - 1][..., :2], atol=1e-5, rtol=1e-5)

    def test_references_are_detached(self) -> None:
        model = _build("detrpose").train()
        calls = _capture_layer_kwargs(model)
        model(_image())
        assert all(not c["kp_reference_points"].requires_grad for c in calls)


def _block_inputs(model: LWDETR, num_queries: int) -> dict:
    """Random, shape-correct inputs for ``TransformerDecoderLayer._detrpose_keypoint_block``.

    Examples:
        >>> sorted(_block_inputs(_build("detrpose"), 4))[:3]
        ['keypoint_class_mask', 'keypoint_pos', 'keypoint_tgt']
    """
    torch.manual_seed(1)
    shapes = torch.tensor([[8, 8], [4, 4]])
    return {
        "tgt": torch.randn(1, num_queries, D),
        "query_pos": torch.randn(1, num_queries, D),
        "keypoint_tgt": torch.randn(1, num_queries, K, D),
        "keypoint_pos": torch.randn(1, num_queries, 1 + K, D),
        "keypoint_class_mask": model.transformer.decoder.keypoint_class_mask,
        "kp_reference_points": torch.rand(1, num_queries * (1 + K), 2, 4) * 0.5 + 0.25,
        "kp_memory": torch.randn(1, 80, D),
        "spatial_shapes": shapes,
        "spatial_shapes_hw": [(8, 8), (4, 4)],
        "level_start_index": torch.tensor([0, 64]),
        "memory_key_padding_mask": torch.zeros(1, 80, dtype=torch.bool),
    }


def _perturb_query(inputs: dict, query: int) -> dict:
    """Copy of ``inputs`` with every per-query input of one query replaced by noise.

    Examples:
        >>> x = {"tgt": torch.zeros(1, 2, 3), "keypoint_tgt": torch.zeros(1, 2, 1, 3)}
        >>> bool(_perturb_query(x, 1)["tgt"][0, 0].abs().sum() == 0)
        True
    """
    out = dict(inputs)
    for key in ("tgt", "query_pos", "keypoint_tgt", "keypoint_pos"):
        if key in out:
            t = out[key].clone()
            t[:, query] = torch.randn_like(t[:, query])
            out[key] = t
    return out


class TestAcrossInstanceAttention:
    def test_groups_do_not_interact_in_training(self) -> None:
        model = _build("detrpose", across_topk=0).train()
        layer = model.transformer.decoder.layers[0]
        inputs = _block_inputs(model, NQ * G)
        tgt_a, kp_a = layer._detrpose_keypoint_block(**inputs)
        tgt_b, kp_b = layer._detrpose_keypoint_block(**_perturb_query(inputs, NQ))  # first query of group 1
        torch.testing.assert_close(kp_a[:, :NQ], kp_b[:, :NQ])
        torch.testing.assert_close(tgt_a[:, :NQ], tgt_b[:, :NQ])

    def test_queries_inside_group_interact(self) -> None:
        model = _build("detrpose", across_topk=0).train()
        layer = model.transformer.decoder.layers[0]
        inputs = _block_inputs(model, NQ * G)
        _, kp_a = layer._detrpose_keypoint_block(**inputs)
        _, kp_b = layer._detrpose_keypoint_block(**_perturb_query(inputs, 1))
        assert not torch.allclose(kp_a[:, 0], kp_b[:, 0])

    def test_queries_outside_topk_are_isolated(self) -> None:
        topk = 3
        model = _build("detrpose", across_topk=topk).eval()
        layer = model.transformer.decoder.layers[0]
        inputs = _block_inputs(model, NQ)
        _, kp_a = layer._detrpose_keypoint_block(**inputs)
        # A non-top-k query does not affect the top-k ones ...
        _, kp_b = layer._detrpose_keypoint_block(**_perturb_query(inputs, topk))
        torch.testing.assert_close(kp_a[:, :topk], kp_b[:, :topk])
        # ... and is itself unaffected by them.
        _, kp_c = layer._detrpose_keypoint_block(**_perturb_query(inputs, 0))
        torch.testing.assert_close(kp_a[:, topk:], kp_c[:, topk:])


class TestGuards:
    def test_requires_two_stage(self) -> None:
        with pytest.raises(ValueError, match="two_stage"):
            _build("detrpose", two_stage=False)

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="keypoint_decoder"):
            _build("detr-pose")

    def test_export_not_supported(self) -> None:
        model = _build("detrpose").eval()
        model.transformer.decoder.export()
        with pytest.raises(NotImplementedError):
            model(_image())
