# ------------------------------------------------------------------------
# COMET3D: detrpose_v2 for RF-DETR
# Copyright (c) 2026 Ashesh Vasalya (avasalya, asheshvasalya@gmail.com). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Switch the training/validation resolution at given epochs (progressive resizing)."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from rfdetr.utilities.logger import get_logger

logger = get_logger()


def resolution_for_epoch(schedule: dict[int, int], epoch: int, default: int) -> int:
    """Return the scheduled resolution for ``epoch`` (latest entry at or before it, else ``default``).

    Examples:
        >>> resolution_for_epoch({10: 672}, 9, 576), resolution_for_epoch({10: 672}, 10, 576)
        (576, 672)
    """
    starts = [start for start in schedule if start <= epoch]
    return schedule[max(starts)] if starts else default


class ResolutionScheduleCallback(Callback):
    """Rebuild the datasets at a new square resolution from scheduled epochs on (e.g. ``{10: 672}``).

    The model is resolution-agnostic (DINOv2 position embeddings are interpolated; multi-scale training already feeds
    varying sizes), so only the datasets change. The switch happens at the end of the previous epoch, after its
    validation, so epoch ``e`` both trains and validates at ``schedule[e]``. On resume, :meth:`on_train_start` applies
    the resolution of the restored epoch. Resolutions must stay divisible by ``patch_size * num_windows``.

    Args:
        schedule: Epoch -> resolution.
        base_resolution: Resolution before the first scheduled epoch.

    Examples:
        >>> ResolutionScheduleCallback({10: 672}, 576).base_resolution
        576
    """

    def __init__(self, schedule: dict[int, int], base_resolution: int) -> None:
        super().__init__()
        self.schedule = {int(k): int(v) for k, v in schedule.items()}
        self.base_resolution = int(base_resolution)

    def _apply(self, trainer: pl.Trainer, pl_module: pl.LightningModule, resolution: int) -> None:
        datamodule: Any = trainer.datamodule
        if datamodule is None or datamodule.model_config.resolution == resolution:
            return
        logger.info("Resolution schedule: %d -> %d px from epoch %d", datamodule.model_config.resolution,
                    resolution, trainer.current_epoch + 1)
        datamodule.model_config.resolution = resolution
        module_config = getattr(pl_module, "model_config", None)
        if module_config is not None and module_config is not datamodule.model_config:
            module_config.resolution = resolution
        datamodule._dataset_train = None
        datamodule._dataset_val = None
        datamodule.setup("fit")
        # Force Lightning to rebuild both dataloaders (and the distributed samplers) at the next epoch start.
        trainer.fit_loop._combined_loader = None
        trainer.fit_loop.epoch_loop.val_loop._combined_loader = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Resume: apply the resolution of the restored epoch."""
        self._apply(trainer, pl_module, resolution_for_epoch(self.schedule, trainer.current_epoch, self.base_resolution))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Switch before the next epoch when it starts a new scheduled resolution."""
        next_epoch = trainer.current_epoch + 1
        if next_epoch in self.schedule:
            self._apply(trainer, pl_module, self.schedule[next_epoch])
