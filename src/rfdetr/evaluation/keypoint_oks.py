# ------------------------------------------------------------------------
# COMET3D: modified for detrpose_v2 (pose stream, part metrics, adaptive LR, FAMO, auto resources)
# Copyright (c) 2026 Ashesh Vasalya (avasalya, asheshvasalya@gmail.com). All Rights Reserved.
# Adapted from RF-DETR (original notice below). Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""OKS keypoint mAP metric backed by :class:`~rfdetr.evaluation.coco_eval.CocoEvaluator`."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import torch

import copy

from faster_coco_eval import COCO

from rfdetr.evaluation.coco_eval import CocoEvaluator, _merge_wholebody_keypoints

#: COCO-WholeBody 133-keypoint parts (start, end) in the concatenated layout (body, foot, face, left/right hand).
WHOLEBODY_PARTS: dict[str, tuple[int, int]] = {
    "body": (0, 17),
    "foot": (17, 23),
    "face": (23, 91),
    "lefthand": (91, 112),
    "righthand": (112, 133),
}


def _part_ground_truth(coco_gt: Any, start: int, end: int) -> COCO:
    """Ground truth restricted to keypoints ``[start, end)`` (WholeBody part fields merged first).

    A person with no labelled keypoint of the part gets ``num_keypoints = 0``, which COCOeval ignores for that part
    (no TP, no FP), as in the official per-part COCO-WholeBody evaluators.

    Args:
        coco_gt: Dataset COCO object.
        start: First keypoint of the part.
        end: One past the last keypoint.

    Returns:
        A new faster-coco-eval ``COCO`` with ``end - start`` keypoints per annotation and category.
    """
    gt = COCO()
    gt.dataset = copy.deepcopy(coco_gt.dataset)
    gt.createIndex()
    _merge_wholebody_keypoints(gt)
    for annotation in gt.dataset.get("annotations", []):
        keypoints = list(annotation.get("keypoints") or [])
        keypoints = keypoints[3 * start : 3 * end] + [0] * max(0, 3 * end - max(len(keypoints), 3 * start))
        annotation["keypoints"] = keypoints[: 3 * (end - start)]
        annotation["num_keypoints"] = sum(1 for v in annotation["keypoints"][2::3] if v > 0)
    for category in gt.dataset.get("categories", []):
        if category.get("keypoints"):
            category["keypoints"] = [f"kp{i}" for i in range(end - start)]
            category["skeleton"] = []
    gt.createIndex()
    label2cat = getattr(coco_gt, "label2cat", None)
    if label2cat is not None:
        gt.label2cat = copy.deepcopy(label2cat)
    return gt


def _slice_keypoints(batch: dict[int, dict[str, Any]], start: int, end: int) -> dict[int, dict[str, Any]]:
    """Copy of a prediction batch whose ``keypoints [N, K, 3]`` keep only ``[start, end)``."""
    sliced = {}
    for image_id, prediction in batch.items():
        prediction = dict(prediction)
        if "keypoints" in prediction and isinstance(prediction["keypoints"], torch.Tensor):
            prediction["keypoints"] = prediction["keypoints"][:, start:end]
        sliced[image_id] = prediction
    return sliced


class OKSKey(str, Enum):
    """Keys returned by :meth:`MetricKeypointOKS.compute`.

    Subclasses :class:`str` so enum members compare equal to their string values
    and can be used interchangeably as dict keys — ``stats[OKSKey.MAP]`` and
    ``stats["map"]`` both work.

    Examples:
        >>> OKSKey.MAP == "map"
        True
        >>> OKSKey.MAP_50.value
        'map_50'
    """

    MAP = "map"
    MAP_50 = "map_50"
    MAP_75 = "map_75"
    MAR = "mar"


def _sanitize_preds(predictions: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return a copy of *predictions* with all tensors detached and moved to CPU.

    Prevents callers from inadvertently retaining CUDA memory or autograd graphs
    between :meth:`MetricKeypointOKS.update` calls.  Non-tensor values are kept as-is.

    Args:
        predictions: Per-image prediction dict mapping ``image_id`` to a dict of
            tensor-valued fields (``boxes``, ``scores``, ``labels``, ``keypoints``).

    Returns:
        New dict with the same structure; every :class:`torch.Tensor` value is
        replaced by its ``.detach().cpu()`` copy.

    Examples:
        >>> import torch
        >>> preds = {1: {"scores": torch.tensor([0.9], device="cpu"), "label": 2}}
        >>> sanitized = _sanitize_preds(preds)
        >>> sanitized[1]["label"]
        2
    """
    return {
        image_id: {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in preds.items()
        }
        for image_id, preds in predictions.items()
    }


# Default ``max_dets`` per image used by :class:`COCOEvalCallback` and
# :class:`MetricKeypointOKS`.  Governs bounding-box and segmentation evaluation
# where max_dets has an effect; for keypoint evaluation the underlying COCO
# evaluator unconditionally overrides maxDets to ``[20]`` regardless of this value
# — this constant has no effect on keypoint AP/AR.
DEFAULT_KEYPOINT_MAX_DETS = 500

# Expected shape of pycocotools _summarizeKps() output.  The keypoint stats array is
# always (10,): AP@50:95 (idx 0), AP@50 (1), AP@75 (2), AP-medium (3), AP-large (4),
# AR@50:95 (5), AR@50 (6), AR@75 (7), AR-medium (8), AR-large (9).
_KPS_STATS_SHAPE = (10,)


class MetricKeypointOKS:
    """OKS keypoint mAP metric backed by CocoEvaluator.

    Plain Python facade over :class:`~rfdetr.evaluation.coco_eval.CocoEvaluator`
    with a :meth:`reset` / :meth:`update` / :meth:`compute` interface that mirrors
    the torchmetrics API shape without subclassing it.

    DDP synchronisation is handled inside :meth:`compute` via
    :meth:`~rfdetr.evaluation.coco_eval.CocoEvaluator.synchronize_between_processes`,
    which uses the repo's pickle-based ``all_gather`` — avoiding the torchmetrics
    deadlock bugs #931 / #449 that affect variable-shape state tensors.

    Supports arbitrary keypoint counts and per-category OKS sigmas through the
    underlying :class:`~rfdetr.evaluation.coco_eval._GroupedKeypointCOCOeval`.

    When TorchMetrics ships production-quality arbitrary-keypoint support (tracked
    in upstream PR #3348), the internals of :meth:`compute` can delegate to
    ``MeanAveragePrecision(iou_type="keypoints", keypoint_format="xyv")`` without
    any change to callers.  Note: when migrating, ``"mar"`` will need remapping to
    ``"mar_<max_dets>"`` as TorchMetrics uses a suffixed key name.

    Args:
        coco_gt: Ground-truth COCO object.  Accepted types: :class:`faster_coco_eval.COCO`
            or any object with a ``.dataset`` dict and optional ``.label2cat`` mapping
            (the duck-typed surface required by :class:`~rfdetr.evaluation.coco_eval.CocoEvaluator`).
        keypoint_oks_sigmas: Per-keypoint OKS sigmas. When ``None``, falls back to
            COCO person sigmas for 17-keypoint datasets or a uniform 0.05 sigma for
            other counts.
        max_dets: Maximum detections per image passed to the underlying
            :class:`~rfdetr.evaluation.coco_eval.CocoEvaluator`.  Defaults to 500.

            Note:
                For keypoint evaluation the underlying COCO evaluator overrides
                ``maxDets`` to ``[20]`` regardless of this value — this parameter
                is forwarded but has no effect on keypoint evaluation.

    Examples:
        >>> from unittest.mock import MagicMock
        >>> metric = MetricKeypointOKS(MagicMock(), max_dets=100)
        >>> metric.has_updates
        False
        >>> metric.reset()  # idempotent on empty state
    """

    def __init__(
        self,
        coco_gt: Any,
        keypoint_oks_sigmas: list[float] | None = None,
        max_dets: int = DEFAULT_KEYPOINT_MAX_DETS,
        part_weights: dict[str, float] | None = None,
        pareto_parts: dict[str, float] | None = None,
    ) -> None:
        self._coco_gt = coco_gt
        self._keypoint_oks_sigmas = keypoint_oks_sigmas
        self._max_dets = max_dets
        # COCO-WholeBody per-part AP (only for 133-keypoint data with 133 sigmas); ground truth built lazily, once.
        self._part_weights = part_weights
        self._part_gts: dict[str, COCO] = {}
        # Pareto gate (part -> noise tolerance in AP units, e.g. {"body": 0.001}; "wholebody" = 133-keypoint AP):
        # map_weighted_pareto rises only when map_weighted rises and no listed part falls below its value at the last
        # accepted epoch minus its tolerance; otherwise it repeats the last accepted value (= no improvement).
        self._pareto_parts = dict(pareto_parts or {})
        self._accepted: dict[str, float] = {}
        # List of per-batch prediction dicts — NOT merged into a single dict.
        # Using a list preserves all predictions when the same image_id appears in
        # multiple batches (e.g. DDP DistributedSampler padding), matching the
        # original CocoEvaluator.update()-per-batch append semantics.
        # Note: tensors are sanitized (detached + CPU) in update() before buffering,
        # so no CUDA graphs or autograd history are retained; for large validation
        # sets this still accumulates ~400 MB of resident CPU memory per rank.
        # Future optimisation: convert to compact COCO result dicts in update() and
        # replay those in compute() instead.
        self._batches: list[dict[int, dict[str, Any]]] = []

    @property
    def has_updates(self) -> bool:
        """Return whether any predictions have been accumulated since last reset.

        Returns:
            ``True`` if :meth:`update` has been called at least once since the
            last :meth:`reset`.

        Examples:
            >>> from unittest.mock import MagicMock
            >>> metric = MetricKeypointOKS(MagicMock())
            >>> metric.has_updates
            False
            >>> metric.update({1: {}})
            >>> metric.has_updates
            True
        """
        return bool(self._batches)

    def reset(self) -> None:
        """Clear accumulated predictions.

        Examples:
            >>> from unittest.mock import MagicMock
            >>> metric = MetricKeypointOKS(MagicMock())
            >>> metric.update({1: {}})
            >>> metric.reset()
            >>> metric.has_updates
            False
        """
        self._batches.clear()

    def update(self, predictions: dict[int, dict[str, Any]]) -> None:
        """Accumulate per-batch predictions.

        Each call appends one batch; predictions are replayed in order inside
        :meth:`compute`.  Predictions for the same ``image_id`` across different
        calls are preserved as separate entries — no overwrite.

        Args:
            predictions: Mapping from ``image_id`` to a prediction dict with keys
                ``boxes`` (``[N, 4]`` xyxy pixel coords), ``scores`` (``[N]``),
                ``labels`` (``[N]`` int), and ``keypoints`` (``[N, K, 3]``
                x/y/confidence in pixel coords). Pass an empty dict for images
                with no predictions.

        Examples:
            >>> from unittest.mock import MagicMock
            >>> metric = MetricKeypointOKS(MagicMock())
            >>> metric.update({1: {}, 2: {}})
            >>> metric.has_updates
            True
        """
        self._batches.append(_sanitize_preds(predictions))

    def compute(self) -> dict[str, float]:
        """Run OKS keypoint evaluation and return metric dict.

        Constructs a fresh :class:`~rfdetr.evaluation.coco_eval.CocoEvaluator`,
        replays all accumulated per-batch predictions in order (matching the
        original per-batch ``CocoEvaluator.update()`` call pattern), synchronises
        across DDP ranks via
        :meth:`~rfdetr.evaluation.coco_eval.CocoEvaluator.synchronize_between_processes`,
        and accumulates COCO keypoint statistics.

        Returns:
            Dict with float values for keys :attr:`OKSKey.MAP` (mAP@50:95),
            :attr:`OKSKey.MAP_50` (AP@50), :attr:`OKSKey.MAP_75` (AP@75),
            and :attr:`OKSKey.MAR` (AR@50:95).  A value of ``-1.0`` indicates
            the statistic was not available (e.g. no predictions matched any ground-truth
            annotation).  Callers should filter ``value < 0`` before logging.

        Examples:
            >>> from unittest.mock import MagicMock, patch
            >>> import numpy as np
            >>> metric = MetricKeypointOKS(MagicMock(), max_dets=500)
            >>> fake_eval = MagicMock()
            >>> fake_eval.coco_eval = {
            ...     "keypoints": MagicMock(stats=np.array([0.5, 0.7, 0.4, -1, -1, 0.6, -1, -1, -1, -1]))
            ... }
            >>> with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=fake_eval):
            ...     metric.update({1: {}})
            ...     result = metric.compute()
            >>> result["map"]
            0.5
            >>> result["map_50"]
            0.7
        """
        evaluator = CocoEvaluator(
            self._coco_gt,
            ["keypoints"],
            max_dets=self._max_dets,
            keypoint_oks_sigmas=self._keypoint_oks_sigmas,
            log_summary=False,
        )
        for batch in self._batches:
            evaluator.update(batch)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        coco_eval_entry: Any = evaluator.coco_eval["keypoints"]
        stats: np.ndarray[Any, Any] = coco_eval_entry.stats
        assert stats.shape == _KPS_STATS_SHAPE, (
            f"Expected coco keypoint stats shape {_KPS_STATS_SHAPE}, got {stats.shape}; "
            "pycocotools _summarizeKps() contract violated — check faster_coco_eval version"
        )
        result: dict[str, float] = {
            OKSKey.MAP: float(stats[0]),
            OKSKey.MAP_50: float(stats[1]),
            OKSKey.MAP_75: float(stats[2]),
            OKSKey.MAR: float(stats[5]),
        }
        result.update(self._compute_parts(result[OKSKey.MAP]))
        return result

    def _compute_parts(self, wholebody_ap: float) -> dict[str, float]:
        """Per-part COCO-WholeBody AP/AR plus the weighted and harmonic means (empty when not configured).

        Returns:
            ``map_<part>``, ``mar_<part>``, ``map_weighted``, ``mar_weighted``, ``map_hmean`` and, with Pareto
            parts, ``map_weighted_pareto``. ``wholebody_ap`` (133-keypoint OKS AP) can be listed as ``wholebody``.
        """
        sigmas = self._keypoint_oks_sigmas
        if not self._part_weights or sigmas is None or len(sigmas) != 133:
            return {}
        result: dict[str, float] = {}
        for part, (start, end) in WHOLEBODY_PARTS.items():
            if part not in self._part_gts:
                self._part_gts[part] = _part_ground_truth(self._coco_gt, start, end)
            evaluator = CocoEvaluator(
                self._part_gts[part],
                ["keypoints"],
                max_dets=self._max_dets,
                keypoint_oks_sigmas=list(sigmas[start:end]),
                log_summary=False,
            )
            for batch in self._batches:
                evaluator.update(_slice_keypoints(batch, start, end))
            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            part_stats = evaluator.coco_eval["keypoints"].stats
            result[f"map_{part}"] = float(part_stats[0])
            result[f"mar_{part}"] = float(part_stats[5])
        total = sum(self._part_weights.get(part, 0.0) for part in WHOLEBODY_PARTS)
        if total > 0:
            for metric in ("map", "mar"):
                result[f"{metric}_weighted"] = sum(
                    self._part_weights.get(part, 0.0) * max(result[f"{metric}_{part}"], 0.0) for part in WHOLEBODY_PARTS
                ) / total
        aps = [max(result[f"map_{part}"], 1e-6) for part in WHOLEBODY_PARTS]
        result["map_hmean"] = len(aps) / sum(1.0 / ap for ap in aps)
        if self._pareto_parts and "map_weighted" in result:
            current = {part: wholebody_ap if part == "wholebody" else result.get(f"map_{part}", -1.0)
                       for part in self._pareto_parts}
            current["weighted"] = result["map_weighted"]
            accepted = self._accepted
            no_part_fell = all(
                current[part] >= accepted[part] - float(tolerance)
                for part, tolerance in self._pareto_parts.items()
                if part in accepted and current[part] >= 0
            )
            if not accepted or (current["weighted"] > accepted["weighted"] and no_part_fell):
                self._accepted = current
            result["map_weighted_pareto"] = self._accepted["weighted"]
        return result
