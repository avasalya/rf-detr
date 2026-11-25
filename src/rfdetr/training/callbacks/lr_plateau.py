# ------------------------------------------------------------------------
# COMET3D: detrpose_v2 for RF-DETR
# Copyright (c) 2026 Ashesh Vasalya (avasalya, asheshvasalya@gmail.com). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Plateau-triggered LR decay for the managed ``adaptive`` (smooth) and ``flat_cosine`` schedules."""

from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class PlateauLRScaleCallback(Callback):
    """Lower the LR scale by ``factor`` after ``patience`` validations without improvement.

    ``adaptive``: the scale moves smoothly (half cosine over ``transition_epochs``). ``flat_cosine``: the multiplier
    of the warmup → flat → cosine curve drops at once. Improvement means the monitored value rose by more than
    ``min_delta`` over the best seen so far. Validations where the metric is absent (``eval_interval``) are skipped.

    Args:
        monitor: Metric key in ``trainer.callback_metrics`` (higher is better).
        patience: Validations without improvement before a drop.
        factor: Multiplier applied per drop.
        min_scale: Floor of the accumulated scale.
        min_delta: Minimum increase that counts as improvement.

    Examples:
        >>> cb = PlateauLRScaleCallback("val/ema_keypoint_map_weighted", patience=4)
        >>> cb.state_dict()["wait"]
        0
    """

    def __init__(
        self,
        monitor: str,
        patience: int,
        factor: float = 0.3,
        min_scale: float = 0.01,
        min_delta: float = 0.001,
        transition_epochs: float = 3.0,
    ) -> None:
        super().__init__()
        self.transition_epochs = float(transition_epochs)
        self.monitor = monitor
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_scale = float(min_scale)
        self.min_delta = float(min_delta)
        self.best = -math.inf
        self.wait = 0

    def _lambdas(self, trainer: pl.Trainer) -> list[tuple[Any, Any]]:
        from rfdetr.training.module_model import AdaptiveDecayLambda, FlatCosineLambda

        found = []
        for config in trainer.lr_scheduler_configs:
            for lr_lambda in getattr(config.scheduler, "lr_lambdas", []):
                if isinstance(lr_lambda, (FlatCosineLambda, AdaptiveDecayLambda)):
                    found.append((config.scheduler, lr_lambda))
        return found

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update the patience counter and drop the LR scale when it runs out."""
        if trainer.sanity_checking or self.patience <= 0:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        current = float(value)
        if current > self.best + self.min_delta:
            self.best = current
            self.wait = 0
            return
        self.wait += 1
        if self.wait < self.patience:
            return
        self.wait = 0
        from rfdetr.training.module_model import AdaptiveDecayLambda

        for scheduler, lr_lambda in self._lambdas(trainer):
            if isinstance(lr_lambda, AdaptiveDecayLambda):
                step = int(scheduler.last_epoch)
                before = lr_lambda.scale(step)
                lr_lambda.begin_decay(
                    step,
                    self.factor,
                    int(self.transition_epochs * lr_lambda.steps_per_epoch),
                    self.min_scale,
                )
                logger.info(
                    "%s has not improved for %d validations (best %.4f): LR scale %.4g -> %.4g over %.1f epochs",
                    self.monitor, self.patience, self.best, before, lr_lambda.scale_to, self.transition_epochs,
                )
                continue
            new_scale = max(self.min_scale, lr_lambda.scale * self.factor)
            if new_scale < lr_lambda.scale:
                logger.info(
                    "%s has not improved for %d validations (best %.4f): LR scale %.4g -> %.4g",
                    self.monitor, self.patience, self.best, lr_lambda.scale, new_scale,
                )
                lr_lambda.scale = new_scale

    def state_dict(self) -> dict[str, Any]:
        """Persist the patience state (the scale itself lives in the scheduler state)."""
        return {"best": self.best, "wait": self.wait}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the patience state."""
        self.best = float(state_dict.get("best", -math.inf))
        self.wait = int(state_dict.get("wait", 0))
