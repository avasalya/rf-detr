# ------------------------------------------------------------------------
# COMET3D: detrpose_v2 for RF-DETR
# Copyright (c) 2026 Ashesh Vasalya (avasalya, asheshvasalya@gmail.com). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Vendored from DETRPose (https://github.com/SebastianJanampa/DETRPose), Apache-2.0:
#   DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
#   Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
#   Modified from DEIM, D-FINE, RT-DETR, GroupPose, ED-Pose, DINO, DN-DETR and Deformable DETR (all Apache-2.0).
# ------------------------------------------------------------------------
"""Original DETRPose pose decoder (``keypoint_decoder: detrpose_v2``), vendored for RF-DETR.

The layer, gate, FDR integral, LQE, decoder loop and contrastive denoising are the original DETRPose code
(``external/DETRPose/src/models/detrpose``). Numerical behaviour is kept; the differences are listed here and are
covered by the parity tests in ``tests/models/test_detrpose_v2_parity.py``:

* ``with_pos_embed`` in the original adds the positional query **in place**. That changes the layer input (the
  previous layer's output) as a side effect, which (a) feeds the positional query into the attention values, the
  residual and the gate, and (b) adds the *next* layer's positional query to ``output_pose_detach`` used by the FDR
  head. The forward semantics of both effects are reproduced here out of place, so the code is also correct in fp32
  autograd and under activation checkpointing (the in-place version only trains under AMP, where the saved tensors
  are cast copies).
* ``nn.MultiheadAttention`` is called with ``need_weights=False``: same output, but PyTorch can use the fused SDPA
  kernel instead of materialising the attention weights.
* ``Integral`` drops a no-op ``reshape(-1, 4)`` that fails when ``batch * queries`` is odd.
* Every layer's poses and scores are returned in eval too (the original stops collecting at the last layer); the final
  outputs are identical, and the intermediate ones feed ``aux_outputs`` / ``scripts/viz_decoder_layers.py``.
* Two-stage selection belongs to RF-DETR (shared with the box stream), so the original ``enc_output`` /
  ``enc_out_class_embed`` / anchor generation are not part of this module; the pose stream receives the selected
  tokens and their anchors. Group-DETR groups are folded into the batch dimension, which keeps across-instance
  attention inside a group exactly.
* CDN takes RF-DETR targets (``keypoints [N, K, 3]`` with ``v`` in ``{-1, 0, 1, 2}``, ``-1`` = not annotated) and uses
  only person targets with at least one labelled keypoint; ``.cuda()`` calls use the target device; the DN pose-centre
  reference is the box centre ``(cx, cy)`` (the original computed ``(cxcy + wh) / 2`` on ``cxcywh`` boxes).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from rfdetr.models.math import MLP

__all__ = [
    "DETRPoseV2Stream",
    "DeformableTransformerDecoderLayer",
    "Gate",
    "Integral",
    "LQE",
    "MSDeformAttn",
    "PoseTransformerDecoder",
    "distance2pose",
    "inverse_sigmoid",
    "prepare_for_cdn",
    "weighting_function",
]


def inverse_sigmoid(x: Tensor, eps: float = 1e-3) -> Tensor:
    """DETRPose ``inverse_sigmoid`` (eps 1e-3, input clamped to ``[0, 1]``).

    Args:
        x: Probabilities.
        eps: Clamp for both ``x`` and ``1 - x``.

    Returns:
        Logits ``log(x / (1 - x))``.

    Examples:
        >>> round(float(inverse_sigmoid(torch.tensor(0.5))), 4)
        0.0
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def weighting_function(reg_max: int, up: Tensor, reg_scale: Tensor, deploy: bool = False) -> Tensor:
    """Non-uniform FDR bin weights ``W(n)`` (D-FINE / DETRPose).

    Args:
        reg_max: Number of bins minus one.
        up: Upper-bound control, ``[1]``.
        reg_scale: Curvature control, ``[1]``.
        deploy: Return a constant ``[1, reg_max + 1]`` tensor (export).

    Returns:
        ``[reg_max + 1]`` bin weights (``[1, reg_max + 1]`` when ``deploy``).

    Examples:
        >>> weighting_function(32, torch.tensor([0.5]), torch.tensor([4.0])).shape
        torch.Size([33])
    """
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.tensor([values], dtype=up.dtype, device=up.device)
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    return torch.cat(values, 0)


def distance2pose(points: Tensor, distance: Tensor, reg_scale: Tensor) -> Tensor:
    """Decode FDR distances into keypoint coordinates: ``xy + distance / reg_scale``.

    Args:
        points: Reference keypoints ``[..., 2]`` (sigmoid image coords).
        distance: Integrated FDR distances ``[..., 2]``.
        reg_scale: FDR scale.

    Returns:
        Keypoints ``[..., 2]``.

    Examples:
        >>> distance2pose(torch.zeros(1, 2), torch.ones(1, 2), torch.tensor(4.0)).tolist()
        [[0.25, 0.25]]
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] + distance[..., 0] / reg_scale
    y1 = points[..., 1] + distance[..., 1] / reg_scale
    return torch.stack([x1, y1], -1)


class Gate(nn.Module):
    """DETRPose gated fusion of the self-attention stream and the cross-attention output."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = float(-math.log((1 - 0.5) / 0.5))
        nn.init.constant_(self.gate.bias, bias)
        nn.init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Return ``LayerNorm(g1 * x1 + g2 * x2)`` with ``g = sigmoid(Linear([x1, x2]))``."""
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """FDR integral ``sum_n softmax(x)_n * W(n)`` over ``reg_max + 1`` bins per coordinate."""

    def __init__(self, reg_max: int = 32) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x: Tensor, project: Tensor) -> Tensor:
        """Integrate ``x [..., C * (reg_max + 1)]`` into ``[..., C]``."""
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device))
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    """Localization-quality estimation: score correction from feature statistics sampled at the predicted keypoints."""

    def __init__(self, topk: int, hidden_dim: int, num_layers: int, num_body_points: int) -> None:
        super().__init__()
        self.k = topk
        self.hidden_dim = hidden_dim
        self.reg_conf = MLP(num_body_points * (topk + 1), hidden_dim, 1, num_layers)
        nn.init.constant_(self.reg_conf.layers[-1].weight.data, 0)
        nn.init.constant_(self.reg_conf.layers[-1].bias.data, 0)
        self.num_body_points = num_body_points

    def forward(self, scores: Tensor, pred_poses: Tensor, feat: Tensor) -> Tensor:
        """Return ``scores + quality`` for ``pred_poses [B, L, K, 2]`` sampled on ``feat [B, C, H, W]``."""
        bs, num_queries = pred_poses.shape[:2]
        pred_poses = pred_poses.reshape(bs, num_queries, self.num_body_points, 2)
        sampling_values = F.grid_sample(
            feat, 2 * pred_poses - 1, mode="bilinear", padding_mode="zeros", align_corners=False
        ).permute(0, 2, 3, 1)
        prob_topk = sampling_values.topk(self.k, dim=-1)[0]
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(bs, num_queries, -1))
        return scores + quality_score


def ms_deform_attn_core_pytorch(
    value: Sequence[Tensor],
    value_spatial_shapes: Sequence[Sequence[int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """DETRPose's pure-PyTorch multi-scale deformable attention core (no value/output projections).

    Args:
        value: Per level ``[N * heads, head_dim, H_l * W_l]``.
        value_spatial_shapes: Per level ``(H_l, W_l)``.
        sampling_locations: ``[N, Lq, heads, levels, points, 2]`` in ``[0, 1]``.
        attention_weights: ``[N, Lq, heads, levels, points]``.

    Returns:
        ``[N, Lq, heads * head_dim]``.
    """
    _, head_dim, _ = value[0].shape
    batch, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1
    sampling_grids = sampling_grids.transpose(1, 2).flatten(0, 1)
    sampling_value_list = []
    for level, (height, width) in enumerate(value_spatial_shapes):
        value_l = value[level].unflatten(2, (int(height), int(width)))
        sampling_grid_l = sampling_grids[:, :, level]
        sampling_value_list.append(
            F.grid_sample(value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False)
        )
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(batch, num_heads * head_dim, num_queries)
    )
    return output.transpose(1, 2)


class MSDeformAttn(nn.Module):
    """DETRPose ``MSDeformAttn``: sampling offsets + attention weights only; values are the raw memory per head."""

    def __init__(
        self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4, use_4D_normalizer: bool = False
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.use_4D_normalizer = use_4D_normalizer
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i % 4 + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if self.n_points % 4 != 0:
            nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Sequence[Tensor],
        input_spatial_shapes: Sequence[Sequence[int]],
    ) -> Tensor:
        """Sample ``value`` around ``reference_points [N, Lq_groups, levels?, T, 2]`` (DETRPose layout).

        Args:
            query: ``[N, Lq, C]``.
            reference_points: ``[N, nq, 1, T, 2]`` (or 4-d boxes); transposed/flattened to ``[N, Lq, 1, 2]``.
            value: Per level ``[N * heads, head_dim, H_l * W_l]``.
            input_spatial_shapes: Per level ``(H_l, W_l)``.

        Returns:
            ``[N, Lq, C]``.
        """
        batch, num_queries, _ = query.shape
        sampling_offsets = self.sampling_offsets(query).view(
            batch, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        reference_points = torch.transpose(reference_points, 2, 3).flatten(1, 2)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(input_spatial_shapes, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.n_levels, 1, 2)
            sampling_locations = (
                reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer.to(query.dtype)
            )
        elif reference_points.shape[-1] == 4:
            if self.use_4D_normalizer:
                shapes = torch.as_tensor(input_spatial_shapes, device=query.device)
                offset_normalizer = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets
                    / offset_normalizer[None, None, None, :, None, :]
                    * reference_points[:, :, None, :, None, 2:]
                    * 0.5
                )
            else:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
                )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}.")
        return ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """DETRPose activation lookup (``relu``/``gelu``/``glu``/``selu``)."""
    table: dict[str, Callable[[Tensor], Tensor]] = {"relu": F.relu, "gelu": F.gelu, "glu": F.glu, "selu": F.selu}
    if activation not in table:
        raise RuntimeError(f"activation should be one of {sorted(table)}, not {activation}.")
    return table[activation]


def _add_kp_pos(tensor: Tensor, pos: Tensor | None) -> Tensor:
    """Out-of-place equivalent of DETRPose ``with_pos_embed``: ``tensor[:, :, -K:] + pos`` (slot 0 unchanged)."""
    if pos is None:
        return tensor
    num_kp = pos.shape[2]
    return torch.cat([tensor[:, :, :-num_kp], tensor[:, :, -num_kp:] + pos], dim=2)


class DeformableTransformerDecoderLayer(nn.Module):
    """Original DETRPose decoder layer: within-instance SA → across-instance SA → deformable CA → Gate → FFN."""

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ) -> None:
        super().__init__()
        self.within_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.within_dropout = nn.Dropout(dropout)
        self.within_norm = nn.LayerNorm(d_model)
        self.across_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.across_dropout = nn.Dropout(dropout)
        self.across_norm = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward_FFN(self, tgt: Tensor) -> Tensor:
        """FFN + residual + LayerNorm (fp16-range clamp as in the original)."""
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm2(tgt.clamp(min=-65504, max=65504))

    def forward(
        self,
        tgt_pose: Tensor,
        tgt_pose_query_pos: Tensor | None,
        tgt_pose_reference_points: Tensor,
        attn_mask: Tensor | None = None,
        memory: Sequence[Tensor] | None = None,
        memory_spatial_shapes: Sequence[Sequence[int]] | None = None,
    ) -> Tensor:
        """Run one layer on ``tgt_pose [B, nq, 1 + K, C]`` (slot 0 = instance token).

        Args:
            tgt_pose: Pose tokens.
            tgt_pose_query_pos: Positional query of the keypoint tokens ``[B, nq, K, C]``.
            tgt_pose_reference_points: ``[B, nq, 1, 1 + K, 2]`` deformable references.
            attn_mask: Optional across-instance mask ``[nq, nq]`` (True = blocked; CDN).
            memory: Per level ``[B * heads, head_dim, H_l * W_l]``.
            memory_spatial_shapes: Per level ``(H_l, W_l)``.

        Returns:
            Updated tokens ``[B, nq, 1 + K, C]``.
        """
        assert memory is not None and memory_spatial_shapes is not None
        bs, nq, num_kpt, d_model = tgt_pose.shape

        # within-instance self-attention; the original's in-place pos add also feeds values and residual.
        tgt_pose = _add_kp_pos(tgt_pose, tgt_pose_query_pos)
        tokens = tgt_pose.flatten(0, 1)
        tgt2 = self.within_attn(tokens, tokens, tokens, need_weights=False)[0].reshape(bs, nq, num_kpt, d_model)
        tgt_pose = tgt_pose + self.within_dropout(tgt2)
        tgt_pose = self.within_norm(tgt_pose)

        # across-instance self-attention (same keypoint index across instances)
        tgt_pose = tgt_pose.transpose(1, 2).flatten(0, 1)
        tgt2_pose = self.across_attn(tgt_pose, tgt_pose, tgt_pose, attn_mask=attn_mask, need_weights=False)[0]
        tgt_pose = tgt_pose + self.across_dropout(tgt2_pose.reshape(bs * num_kpt, nq, d_model))
        tgt_pose = self.across_norm(tgt_pose).reshape(bs, num_kpt, nq, d_model).transpose(1, 2)

        # deformable cross-attention; the original's in-place pos add also feeds the gate.
        tgt_pose = _add_kp_pos(tgt_pose, tgt_pose_query_pos)
        tgt2_pose = self.cross_attn(
            tgt_pose.flatten(1, 2), tgt_pose_reference_points, memory, memory_spatial_shapes
        ).reshape(bs, nq, num_kpt, d_model)
        tgt_pose = self.gateway(tgt_pose, self.dropout1(tgt2_pose))
        return self.forward_FFN(tgt_pose)


def _get_clones(module: nn.Module, num_clones: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_clones)])


class PoseTransformerDecoder(nn.Module):
    """Original DETRPose ``TransformerDecoder``: iterative FDR refinement of ``[centre + K]`` pose references."""

    def __init__(
        self,
        decoder_layer: DeformableTransformerDecoderLayer,
        num_layers: int,
        hidden_dim: int = 256,
        num_body_points: int = 17,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers) if num_layers > 0 else nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_body_points = num_body_points
        self.half_pose_ref_point_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.eval_idx = num_layers - 1
        self.activation_checkpointing = activation_checkpointing
        dim_t = torch.arange(hidden_dim // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (hidden_dim // 2))
        self.register_buffer("dim_t", dim_t)
        self.scale = 2 * math.pi

    def sine_embedding(self, pos_tensor: Tensor) -> Tensor:
        """Sine embedding of ``[..., 2]`` (or ``[..., 4]``) positions, ``(y, x[, w, h])`` order as in the original."""
        x_embed = pos_tensor[..., 0:1] * self.scale
        y_embed = pos_tensor[..., 1:2] * self.scale
        pos_x = x_embed / self.dim_t
        pos_y = y_embed / self.dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        if pos_tensor.size(-1) == 2:
            return torch.cat((pos_y, pos_x), dim=3)
        if pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[..., 2:3] * self.scale
            pos_w = w_embed / self.dim_t
            pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=4).flatten(3)
            h_embed = pos_tensor[..., 3:4] * self.scale
            pos_h = h_embed / self.dim_t
            pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=4).flatten(3)
            return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=3)
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")

    def _run_layer(
        self,
        layer: nn.Module,
        output: Tensor,
        pose_query_pos: Tensor,
        refpoint_pose_input: Tensor,
        attn_mask: Tensor | None,
        memory: Sequence[Tensor],
        spatial_shapes: Sequence[Sequence[int]],
    ) -> Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():

            def run(out: Tensor, pos: Tensor, ref: Tensor, *mem: Tensor) -> Tensor:
                return layer(out, pos, ref, attn_mask=attn_mask, memory=list(mem), memory_spatial_shapes=spatial_shapes)

            return checkpoint(run, output, pose_query_pos, refpoint_pose_input, *memory, use_reentrant=False)
        return layer(
            output,
            pose_query_pos,
            refpoint_pose_input,
            attn_mask=attn_mask,
            memory=memory,
            memory_spatial_shapes=spatial_shapes,
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Sequence[Tensor],
        refpoints_sigmoid: Tensor,
        pre_pose_head: nn.Module,
        pose_head: nn.ModuleList,
        class_head: nn.ModuleList,
        lqe_head: nn.ModuleList,
        feat_lqe: Tensor,
        integral: Integral,
        reg_scale: Tensor,
        project: Tensor,
        attn_mask: Tensor | None = None,
        spatial_shapes: Sequence[Sequence[int]] | None = None,
    ) -> dict[str, Tensor]:
        """Run the decoder loop; returns every layer's poses, logits, FDR corners, references and hidden tokens.

        Args:
            tgt: Initial tokens ``[B, nq, 1 + K, C]``.
            memory: Per level ``[B * heads, head_dim, H_l * W_l]``.
            refpoints_sigmoid: Initial ``[centre + K]`` references ``[B, nq, 1 + K, 2]``.
            pre_pose_head: Layer-0 pose head (``pre_pose_embed``).
            pose_head: Per-layer FDR heads.
            class_head: Per-layer instance-token classifiers.
            lqe_head: Per-layer LQE modules.
            feat_lqe: Feature map sampled by LQE ``[B, C, H, W]``.
            integral: FDR integral.
            reg_scale: FDR scale.
            project: FDR bin weights.
            attn_mask: Across-instance attention mask (CDN).
            spatial_shapes: Per level ``(H_l, W_l)``.

        Returns:
            ``poses [L, B, nq, K, 2]``, ``logits [L, B, nq, n_cls]``, ``corners``, ``refs``, ``hs [L, B, nq, K, C]``,
            ``pre_poses [B, nq, K, 2]``, ``pre_logits [B, nq, n_cls]``.
        """
        assert spatial_shapes is not None
        output = tgt
        refpoint_pose = refpoints_sigmoid
        output_pose_detach: Tensor | int = 0
        pred_corners_undetach: Tensor | int = 0
        dec_out_poses, dec_out_logits, dec_out_refs, dec_out_pred_corners, dec_out_hs = [], [], [], [], []
        pre_poses = pre_scores = ref_pose_initial = None

        for layer_id, layer in enumerate(self.layers):
            refpoint_pose_input = refpoint_pose[:, :, None]
            refpoint_only_pose = refpoint_pose[:, :, 1:]
            pose_query_sine_embed = self.sine_embedding(refpoint_only_pose)
            pose_query_pos = self.half_pose_ref_point_head(pose_query_sine_embed)
            if layer_id > 0:
                # The original adds this layer's positional query in place to its input, which aliases the previous
                # layer's output and therefore ``output_pose_detach``; reproduce that side effect explicitly.
                output_pose_detach = output_pose_detach + (pose_query_pos.detach() if self.training else pose_query_pos)

            output = self._run_layer(
                layer, output, pose_query_pos, refpoint_pose_input, attn_mask, memory, spatial_shapes
            )
            output_pose = output[:, :, 1:]
            output_instance = output[:, :, 0]

            if layer_id == 0:
                pre_poses = torch.sigmoid(pre_pose_head(output_pose) + inverse_sigmoid(refpoint_only_pose))
                pre_scores = class_head[0](output_instance)
                ref_pose_initial = pre_poses.detach()
            assert ref_pose_initial is not None

            pred_corners = pose_head[layer_id](output_pose + output_pose_detach) + pred_corners_undetach
            refpoint_pose_without_center = distance2pose(ref_pose_initial, integral(pred_corners, project), reg_scale)
            refpoint_center_pose = torch.mean(refpoint_pose_without_center, dim=2, keepdim=True)
            refpoint_pose = torch.cat([refpoint_center_pose, refpoint_pose_without_center], dim=2)

            score = class_head[layer_id](output_instance)
            dec_out_logits.append(lqe_head[layer_id](score, refpoint_pose_without_center, feat_lqe))
            dec_out_poses.append(refpoint_pose_without_center)
            dec_out_pred_corners.append(pred_corners)
            dec_out_refs.append(ref_pose_initial)
            dec_out_hs.append(output_pose)

            pred_corners_undetach = pred_corners
            if self.training:
                refpoint_pose = refpoint_pose.detach()
                output_pose_detach = output_pose.detach()
            else:
                output_pose_detach = output_pose

        assert pre_poses is not None and pre_scores is not None
        return {
            "poses": torch.stack(dec_out_poses),
            "logits": torch.stack(dec_out_logits),
            "corners": torch.stack(dec_out_pred_corners),
            "refs": torch.stack(dec_out_refs),
            "hs": torch.stack(dec_out_hs),
            "pre_poses": pre_poses,
            "pre_logits": pre_scores,
        }


def prepare_for_cdn(
    targets: list[dict[str, Tensor]],
    dn_number: int,
    label_noise_ratio: float,
    num_queries: int,
    num_classes: int,
    num_keypoints: int,
    hidden_dim: int,
    label_enc: nn.Embedding,
    pose_enc: nn.Embedding,
    img_dim: tuple[int, int],
    sigmas: Tensor,
    person_label: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]] | None:
    """Contrastive pose denoising queries (DETRPose ``prepare_for_cdn``) from RF-DETR targets.

    Args:
        targets: RF-DETR targets: ``labels [n]``, ``boxes [n, 4]`` (cxcywh, normalised), ``keypoints [n, K, 3]``
            (normalised xy, ``v`` in ``{-1, 0, 1, 2}``).
        dn_number: Denoising query budget (original 20; doubled for positive/negative pairs).
        label_noise_ratio: Probability ×0.5 of flipping a DN query's label.
        num_queries: Matching queries that follow the DN block in the sequence.
        num_classes: Label-embedding classes (``label_enc`` has ``num_classes + 1`` rows).
        num_keypoints: ``K``.
        hidden_dim: Token width.
        label_enc: Instance-token label embedding.
        pose_enc: Keypoint-token embedding.
        img_dim: Input ``(H, W)`` in pixels (noise scale).
        sigmas: Per-keypoint OKS sigmas ``[K]``.
        person_label: Target label of the keypoint (person) class.
        device: Device for the generated tensors.

    Returns:
        ``None`` when no image has a usable person, else ``(query_tokens [B, pad, 1 + K, C], query_refs_logit
        [B, pad, 1 + K, 2], attn_mask [pad + nq, pad + nq], dn_meta)``. ``dn_meta`` holds ``pad_size``,
        ``num_dn_group``, ``single_pad`` and ``known_targets`` (per image, the target indices used, in DN order).
    """
    dn_number = dn_number * 2
    known_targets = []
    for target in targets:
        labels = target["labels"]
        kps = target.get("keypoints")
        if kps is None or kps.numel() == 0:
            known_targets.append(torch.zeros(0, dtype=torch.long, device=labels.device))
            continue
        usable = (labels == person_label) & (kps[..., 2] > 0).any(dim=-1)
        known_targets.append(torch.nonzero(usable).view(-1))
    known_num = [int(k.numel()) for k in known_targets]
    batch_size = len(targets)
    if batch_size == 0 or max(known_num) == 0:
        return None

    dn_number = dn_number // (int(max(known_num) * 2))
    dn_number = 1 if dn_number == 0 else dn_number

    labels = torch.cat([t["labels"][k] for t, k in zip(targets, known_targets)]).to(device)
    batch_idx = torch.cat(
        [torch.full_like(k, i) for i, k in enumerate(known_targets)]
    ).to(device)
    known_labels = labels.repeat(2 * dn_number, 1).view(-1)
    known_labels_expaned = known_labels.clone()
    known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)

    if label_noise_ratio > 0:
        p = torch.rand_like(known_labels_expaned.float())
        chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)
        new_label = torch.randint_like(chosen_indice, 0, num_classes)
        known_labels_expaned.scatter_(0, chosen_indice, new_label)

    boxes = torch.cat([t["boxes"][k] for t, k in zip(targets, known_targets)]).to(device)
    keypoints = torch.cat([t["keypoints"][k] for t, k in zip(targets, known_targets)]).to(device)
    areas = boxes[:, 2] * boxes[:, 3] * 0.53  # original fallback when no segmentation area is given
    poses = torch.cat([boxes[:, None, :2], keypoints[..., :2]], dim=1)  # [n, 1 + K, 2]
    non_viz = keypoints[..., 2] <= 0
    non_viz = torch.cat((torch.ones_like(non_viz[:, 0:1]), non_viz), dim=1)
    center_sigma = sigmas.new_tensor([0.1])
    variances = ((2 * torch.cat([center_sigma, sigmas.to(device)])) ** 2)[None, :, None]

    known_poses = poses.repeat(2 * dn_number, 1, 1)
    known_areas = areas.repeat(2 * dn_number)[..., None, None] * img_dim[0] * img_dim[1]
    known_non_viz = non_viz.repeat(2 * dn_number, 1)

    single_pad = int(max(known_num))
    pad_size = int(single_pad * 2 * dn_number)
    num_poses = len(poses)
    positive_idx = torch.arange(num_poses, device=device).long().unsqueeze(0).repeat(dn_number, 1)
    positive_idx += (torch.arange(dn_number, device=device) * num_poses * 2).long().unsqueeze(1)
    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + num_poses

    eps = float(np.finfo("float32").eps)
    rand_vector = F.normalize(torch.rand_like(known_poses), dim=-1)
    rand_alpha = torch.zeros_like(known_poses[..., :1]).uniform_(-np.log(1), -np.log(0.5))
    rand_alpha[negative_idx] = rand_alpha[negative_idx].uniform_(-np.log(0.5), -np.log(0.1))
    rand_alpha = rand_alpha * 2 * (known_areas + eps) * variances
    rand_alpha = torch.sqrt(rand_alpha) / max(img_dim)
    rand_alpha[known_non_viz] = 0.0
    known_poses_expand = known_poses + rand_alpha * rand_vector

    input_label_embed = label_enc(known_labels_expaned.long())
    input_label_pose_embed = pose_enc.weight[None].repeat(known_poses_expand.size(0), 1, 1)
    input_label_embed = torch.cat([input_label_embed.unsqueeze(1), input_label_pose_embed], dim=1).flatten(1)
    input_pose_embed = inverse_sigmoid(known_poses_expand)

    input_query_label = torch.zeros(batch_size, pad_size, hidden_dim * (num_keypoints + 1), device=device)
    input_query_label = input_query_label.to(input_label_embed.dtype)
    input_query_pose = torch.zeros(batch_size, pad_size, num_keypoints + 1, 2, device=device)
    map_known_indice = torch.cat([torch.arange(num, device=device) for num in known_num]).long()
    map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)])
    input_query_label = input_query_label.index_put((known_bid.long(), map_known_indice), input_label_embed)
    input_query_pose = input_query_pose.index_put((known_bid.long(), map_known_indice), input_pose_embed)

    tgt_size = pad_size + num_queries
    attn_mask = torch.zeros(tgt_size, tgt_size, dtype=torch.bool, device=device)
    attn_mask[pad_size:, :pad_size] = True  # matching queries cannot see the reconstruction
    for i in range(dn_number):  # reconstruction groups cannot see each other
        if i == 0:
            attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True
        if i == dn_number - 1:
            attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2] = True
        else:
            attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True
            attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i] = True

    dn_meta = {
        "pad_size": pad_size,
        "num_dn_group": dn_number,
        "single_pad": single_pad,
        "known_targets": known_targets,
    }
    return input_query_label.unflatten(-1, (-1, hidden_dim)), input_query_pose, attn_mask, dn_meta


class DETRPoseV2Stream(nn.Module):
    """Pose-only query stream of DETRPose (6 layers, FDR, LQE, CDN), spawned from RF-DETR's two-stage selection.

    Parameter names follow the original ``Transformer`` so shape-compatible parts of an official DETRPose checkpoint
    can be loaded (``decoder.*``, ``keypoint_embedding``, ``instance_embedding``, ``class_embed``, ``pose_embed``, …).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_body_points: int = 133,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dec_n_points: int = 4,
        num_feature_levels: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        num_queries: int = 100,
        learnable_tgt_init: bool = True,
        num_classes: int = 1,
        reg_max: int = 32,
        reg_scale: float = 4.0,
        dn_number: int = 20,
        dn_label_noise_ratio: float = 0.5,
        extra_heads: bool = True,
        activation_checkpointing: bool = False,
        oks_sigmas: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_body_points = num_body_points
        self.num_decoder_layers = num_decoder_layers
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.dn_number = dn_number
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.reg_max = reg_max

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points
        )
        self.decoder = PoseTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            hidden_dim=hidden_dim,
            num_body_points=num_body_points,
            activation_checkpointing=activation_checkpointing,
        )
        self.keypoint_embedding = nn.Embedding(num_body_points, hidden_dim)
        self.instance_embedding = nn.Embedding(1, hidden_dim)
        self.learnable_tgt_init = learnable_tgt_init
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim) if learnable_tgt_init else None
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim)
        self.pose_enc = nn.Embedding(num_body_points, hidden_dim)
        self._reset_parameters()

        _class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        _class_embed.bias.data = torch.ones(num_classes) * -math.log((1 - prior_prob) / prior_prob)
        _pre_point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pre_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pre_point_embed.layers[-1].bias.data, 0)
        _point_embed = MLP(hidden_dim, hidden_dim, 2 * (reg_max + 1), 3)
        nn.init.constant_(_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_point_embed.layers[-1].bias.data, 0)
        _lqe_embed = LQE(4, 256, 2, num_body_points)
        self.class_embed = nn.ModuleList([copy.deepcopy(_class_embed) for _ in range(num_decoder_layers)])
        self.pose_embed = nn.ModuleList([copy.deepcopy(_point_embed) for _ in range(num_decoder_layers)])
        self.lqe_embed = nn.ModuleList([copy.deepcopy(_lqe_embed) for _ in range(num_decoder_layers)])
        self.pre_pose_embed = _pre_point_embed
        self.integral = Integral(reg_max)
        self.up = nn.Parameter(torch.tensor([1 / 2]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        # Two-stage initial pose from the selected encoder token (original ``enc_pose_embed``).
        self.enc_pose_embed = MLP(hidden_dim, 2 * hidden_dim, 2 * num_body_points, 4)
        nn.init.constant_(self.enc_pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.enc_pose_embed.layers[-1].bias.data, 0)

        # RF-DETR keypoint channels beyond xy (findable, visible, 3 precision-Cholesky values, class contribution).
        self.extra_embed: MLP | None = None
        if extra_heads:
            self.extra_embed = MLP(hidden_dim, hidden_dim, 6, 3)
            nn.init.constant_(self.extra_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.extra_embed.layers[-1].bias.data, 0)

        sigmas = torch.tensor(oks_sigmas if oks_sigmas is not None else [0.05] * num_body_points, dtype=torch.float32)
        if sigmas.numel() != num_body_points:
            raise ValueError(f"oks_sigmas has {sigmas.numel()} values, expected {num_body_points}")
        self.register_buffer("oks_sigmas", sigmas, persistent=False)

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def _memory_values(self, levels: Sequence[Tensor]) -> tuple[list[Tensor], list[list[int]]]:
        """Flatten ``levels [B, C, H, W]`` to the original per-level ``[B * heads, head_dim, H * W]`` values."""
        spatial_shapes = [[int(feat.shape[-2]), int(feat.shape[-1])] for feat in levels]
        split_sizes = [h * w for h, w in spatial_shapes]
        memory = torch.cat([feat.flatten(2).transpose(1, 2) for feat in levels], dim=1)
        value = memory.unflatten(2, (self.nhead, -1)).permute(0, 2, 3, 1).flatten(0, 1).split(split_sizes, dim=-1)
        return list(value), spatial_shapes

    def initial_poses(self, topk_memory: Tensor, topk_anchor_xy: Tensor) -> Tensor:
        """Two-stage initial pose ``sigmoid(enc_pose_embed(token) + inverse_sigmoid(anchor))`` ``[B, N, K, 2]``.

        Args:
            topk_memory: Encoder tokens ``[B, N, C]``.
            topk_anchor_xy: Anchors ``[B, N, 2]`` (sigmoid image coords).

        Returns:
            Initial keypoints in sigmoid image coordinates.
        """
        bs, nq = topk_memory.shape[:2]
        delta = self.enc_pose_embed(topk_memory).reshape(bs, nq, self.num_body_points, 2)
        return torch.sigmoid(delta + inverse_sigmoid(topk_anchor_xy).unsqueeze(-2))

    def initial_queries(self, topk_memory: Tensor, topk_anchor_xy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Original query construction: ``slot 0 = instance_embedding``, ``kp_i = keypoint_embedding_i + tgt``.

        Args:
            topk_memory: Selected encoder tokens ``[B, M, C]``.
            topk_anchor_xy: Their anchors ``[B, M, 2]`` (sigmoid image coords).

        Returns:
            ``(tgt [B, M, 1 + K, C], refs [B, M, 1 + K, 2] detached, enc_poses [B, M, K, 2])``.
        """
        bs, nq = topk_memory.shape[:2]
        enc_poses = self.initial_poses(topk_memory, topk_anchor_xy)
        enc_center = torch.mean(enc_poses, dim=2, keepdim=True)
        refs = torch.cat([enc_center, enc_poses], dim=2).detach()
        if self.tgt_embed is not None:
            if nq > self.tgt_embed.num_embeddings:
                raise ValueError(f"{nq} pose queries exceed tgt_embed rows ({self.tgt_embed.num_embeddings})")
            tgt = self.tgt_embed.weight[:nq].unsqueeze(0).expand(bs, -1, -1).unsqueeze(-2)
        else:
            tgt = topk_memory.detach().unsqueeze(-2)
        tgt_pose = self.keypoint_embedding.weight[None, None].expand(bs, nq, -1, -1) + tgt
        tgt_global = self.instance_embedding.weight[None, None].expand(bs, nq, -1, -1)
        return torch.cat([tgt_global, tgt_pose], dim=2), refs, enc_poses

    def _decode(
        self,
        tgt: Tensor,
        refs: Tensor,
        levels: Sequence[Tensor],
        attn_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        value, spatial_shapes = self._memory_values(levels)
        project = weighting_function(self.reg_max, self.up, self.reg_scale)
        out = self.decoder(
            tgt=tgt,
            memory=value,
            refpoints_sigmoid=refs,
            spatial_shapes=spatial_shapes,
            attn_mask=attn_mask,
            pre_pose_head=self.pre_pose_embed,
            pose_head=self.pose_embed,
            class_head=self.class_embed,
            lqe_head=self.lqe_embed,
            feat_lqe=levels[0],
            reg_scale=self.reg_scale,
            integral=self.integral,
            project=project,
        )
        if self.extra_embed is not None:
            out["extra"] = self.extra_embed(out["hs"])
        return out

    def forward(
        self,
        levels: Sequence[Tensor],
        topk_memory: Tensor,
        topk_anchor_xy: Tensor,
        groups: int = 1,
        targets: list[dict[str, Tensor]] | None = None,
        image_size: tuple[int, int] | None = None,
        person_label: int = 0,
    ) -> dict[str, Any]:
        """Run the pose stream for every group's candidates, plus CDN when training with targets.

        Args:
            levels: Pose-stream feature maps ``[B, C, H_l, W_l]`` (highest resolution first; LQE samples it).
            topk_memory: Candidate tokens ``[B, G, M, C]`` flattened to ``[B * G, M, C]`` (batch-major).
            topk_anchor_xy: Candidate anchors ``[B * G, M, 2]``.
            groups: ``G``; groups are folded into the batch so across-instance attention stays within a group.
            targets: Training targets (enables CDN).
            image_size: Input ``(H, W)`` in pixels (CDN noise scale).
            person_label: Target label of the person class.

        Returns:
            Decoder outputs for ``[B * G, M]`` plus ``enc_poses``; ``dn`` (outputs over the DN slots and ``dn_meta``)
            when CDN ran.
        """
        tgt, refs, enc_poses = self.initial_queries(topk_memory, topk_anchor_xy)
        group_levels = [feat.repeat_interleave(groups, dim=0) for feat in levels] if groups > 1 else list(levels)
        out = self._decode(tgt, refs, group_levels, None)
        out["enc_poses"] = enc_poses

        if self.training and targets is not None and self.dn_number > 0:
            assert image_size is not None
            dn = prepare_for_cdn(
                targets,
                self.dn_number,
                self.dn_label_noise_ratio,
                num_queries=tgt.shape[1],
                num_classes=self.num_classes,
                num_keypoints=self.num_body_points,
                hidden_dim=self.hidden_dim,
                label_enc=self.label_enc,
                pose_enc=self.pose_enc,
                img_dim=image_size,
                sigmas=self.oks_sigmas,
                person_label=person_label,
                device=tgt.device,
            )
            if dn is not None:
                dn_tgt, dn_refs_logit, attn_mask, dn_meta = dn
                pad = dn_meta["pad_size"]
                bs = len(targets)
                # DN queries join group 0's matching queries, which cannot see them (original sequence layout).
                group0 = slice(0, None, groups)
                tgt_dn = torch.cat([dn_tgt.to(tgt.dtype), tgt[group0]], dim=1)
                refs_dn = torch.cat([dn_refs_logit.sigmoid().to(refs.dtype), refs[group0]], dim=1)
                dn_out = self._decode(tgt_dn, refs_dn, list(levels), attn_mask)
                # Layer-stacked outputs are [L, B, nq, ...]; pre_* are [B, nq, ...]. Keep only the DN slots.
                dn_split = {
                    key: value[:, :pad] if key.startswith("pre_") else value[:, :, :pad]
                    for key, value in dn_out.items()
                }
                assert dn_split["poses"].shape[1] == bs and dn_split["poses"].shape[2] == pad
                out["dn"] = {**dn_split, "dn_meta": dn_meta}
        return out


def keypoint_oks(
    pred_xy: Tensor,
    target: Tensor,
    areas: Tensor,
    sigmas: Tensor,
    keypoint_weights: Tensor | None = None,
    pairwise: bool = False,
) -> Tensor:
    """DETRPose OKS (``oks_overlaps``) on normalised coordinates, optionally part-weighted.

    ``OKS = sum_k w_k v_k exp(-d_k^2 / (2 * area * (2 sigma_k)^2)) / sum_k w_k v_k`` over labelled keypoints
    (``v > 0``; ``w_k = 1`` reproduces the original). Targets without a labelled keypoint get OKS 0.

    Args:
        pred_xy: ``[N, K, 2]`` (or ``[P, K, 2]`` when ``pairwise``).
        target: ``[N, K, 3]`` (or ``[T, K, 3]``) with visibility in channel 2.
        areas: ``[N]`` (or ``[T]``) normalised object areas.
        sigmas: ``[K]`` OKS sigmas.
        keypoint_weights: Optional ``[K]`` per-keypoint weights (e.g. per-part shares).
        pairwise: Return the ``[P, T]`` matrix instead of the matched ``[N]`` vector.

    Returns:
        OKS in ``[0, 1]``.

    Examples:
        >>> t = torch.tensor([[[0.5, 0.5, 2.0], [0.2, 0.2, 0.0]]])
        >>> round(float(keypoint_oks(t[..., :2], t, torch.tensor([0.1]), torch.tensor([0.05, 0.05]))), 4)
        1.0
    """
    variances = (sigmas.to(device=pred_xy.device, dtype=pred_xy.dtype) * 2) ** 2
    target = target.to(pred_xy.device)
    areas = areas.to(pred_xy.device)
    valid = (target[..., 2] > 0).to(pred_xy.dtype)
    if keypoint_weights is not None:
        valid = valid * keypoint_weights.to(device=pred_xy.device, dtype=pred_xy.dtype)
    if pairwise:
        d2 = (pred_xy[:, None, :, :] - target[None, :, :, :2].to(pred_xy.dtype)).pow(2).sum(-1)  # [P, T, K]
        e = torch.exp(-d2 / (areas[None, :, None].to(pred_xy.dtype) * variances * 2))
        return (e * valid[None]).sum(-1) / (valid.sum(-1)[None] + 1e-6)
    d2 = (pred_xy - target[..., :2].to(pred_xy.dtype)).pow(2).sum(-1)
    e = torch.exp(-d2 / (areas[:, None].to(pred_xy.dtype) * variances * 2))
    return (e * valid).sum(-1) / (valid.sum(-1) + 1e-6)


def varifocal_pose_loss(logits: Tensor, target_score: Tensor, positive: Tensor, valid: Tensor,
                        alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    """DETRPose ``loss_vfl`` for one person logit per query, summed over ``valid`` queries.

    Args:
        logits: ``[...]`` pose-score logits.
        target_score: ``[...]`` IoU-like targets (OKS at positives, 0 elsewhere).
        positive: ``[...]`` matched person queries.
        valid: ``[...]`` queries that take part (ignored ones contribute nothing).
        alpha: Focal alpha.
        gamma: Focal gamma.

    Returns:
        Summed loss (normalise by the number of targets outside).

    Examples:
        >>> loss = varifocal_pose_loss(torch.zeros(3), torch.tensor([0.8, 0.0, 0.0]),
        ...                            torch.tensor([True, False, False]), torch.ones(3, dtype=torch.bool))
        >>> bool(loss > 0)
        True
    """
    pos = positive.to(logits.dtype)
    pred_score = torch.sigmoid(logits).detach()
    weight = alpha * pred_score.pow(gamma) * (1 - pos) + target_score.to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target_score.to(logits.dtype), weight=weight, reduction="none")
    return (loss * valid.to(logits.dtype)).sum()
