# ------------------------------------------------------------------------
# COMET3D: detrpose_v2 for RF-DETR
# Copyright (c) 2026 Ashesh Vasalya (avasalya, asheshvasalya@gmail.com). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The resolution switch stops the old persistent DataLoader workers before new datasets are built."""

from __future__ import annotations

import torch
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset

from rfdetr.training.callbacks.resolution_schedule import shutdown_workers


def _started_loader() -> DataLoader:
    """A DataLoader with two persistent workers, iterated once so its workers are running.

    Examples:
        >>> loader = _started_loader()
        >>> len(loader._iterator._workers)
        2
        >>> shutdown_workers(CombinedLoader([loader]))
        1
    """
    loader = DataLoader(TensorDataset(torch.arange(8.0)), batch_size=2, num_workers=2, persistent_workers=True)
    for _ in loader:
        pass
    return loader


class TestShutdownWorkers:
    def test_persistent_workers_stopped(self) -> None:
        loader = _started_loader()
        workers = list(loader._iterator._workers)
        assert all(w.is_alive() for w in workers)
        shutdown_workers(CombinedLoader([loader], mode="sequential"))
        for w in workers:
            w.join(timeout=10)
        assert not any(w.is_alive() for w in workers)

    def test_iterator_cleared_so_next_epoch_starts_fresh_workers(self) -> None:
        loader = _started_loader()
        shutdown_workers(CombinedLoader([loader], mode="sequential"))
        assert loader._iterator is None
        assert [b[0].tolist() for b in loader] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]

    def test_none_and_single_process_loaders(self) -> None:
        assert shutdown_workers(None) == 0
        loader = DataLoader(TensorDataset(torch.arange(4.0)), batch_size=2)
        list(loader)
        assert shutdown_workers(CombinedLoader([loader], mode="sequential")) == 0
