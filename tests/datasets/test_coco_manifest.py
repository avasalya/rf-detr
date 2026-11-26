# ------------------------------------------------------------------------
# COMET3D: tests for per-source train annotation manifests
# Copyright (c) 2026 Ashesh Vasalya (avasalya, asheshvasalya@gmail.com). All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
"""Tests for :mod:`rfdetr.datasets.coco_manifest` and the manifest path of :class:`CocoDetection`."""

import json
from pathlib import Path

import pytest

from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.coco_manifest import load_manifest_coco

CATEGORIES = [{"id": 1, "name": "person"}]


def write_source(directory: Path, name: str, base: int, n_images: int) -> dict:
    """Write one per-source COCO file with ids from ``base`` and return its manifest entry.

    Examples:
        >>> import tempfile
        >>> entry = write_source(Path(tempfile.mkdtemp()), "a", 10, 2)
        >>> entry["images"], entry["annotations"]
        (2, 2)
    """
    images = [{"id": base + i, "file_name": f"{name}/{i}.jpg", "width": 4, "height": 4} for i in range(1, n_images + 1)]
    annotations = [
        {"id": base + i, "image_id": base + i, "category_id": 1, "bbox": [0, 0, 2, 2], "area": 4.0, "iscrowd": 0}
        for i in range(1, n_images + 1)
    ]
    (directory / f"{name}.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES})
    )
    return {"name": name, "file": f"{name}.json", "images": n_images, "annotations": n_images}


def write_manifest(directory: Path, entries: list) -> Path:
    """Write ``manifest.json`` for ``entries`` and return its path.

    Examples:
        >>> import tempfile
        >>> write_manifest(Path(tempfile.mkdtemp()), []).name
        'manifest.json'
    """
    path = directory / "manifest.json"
    path.write_text(json.dumps({"format": 1, "id_block": 10, "categories": CATEGORIES, "sources": entries}))
    return path


class TestLoadManifestCoco:
    def test_concatenates_sources_in_manifest_order(self, tmp_path: Path) -> None:
        manifest = write_manifest(tmp_path, [write_source(tmp_path, "a", 10, 2), write_source(tmp_path, "b", 20, 3)])
        coco = load_manifest_coco(manifest)
        assert [image["id"] for image in coco.dataset["images"]] == [11, 12, 21, 22, 23]
        assert coco.getAnnIds(imgIds=[22]) == [22]

    def test_count_mismatch_raises(self, tmp_path: Path) -> None:
        entry = write_source(tmp_path, "a", 10, 2)
        entry["images"] = 3
        with pytest.raises(ValueError, match="manifest says"):
            load_manifest_coco(write_manifest(tmp_path, [entry]))

    def test_duplicate_ids_raise(self, tmp_path: Path) -> None:
        entries = [write_source(tmp_path, "a", 10, 2), write_source(tmp_path, "b", 10, 2)]
        with pytest.raises(ValueError, match="duplicate image ids"):
            load_manifest_coco(write_manifest(tmp_path, entries))


def test_coco_detection_reads_manifest(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, [write_source(tmp_path, "a", 10, 2), write_source(tmp_path, "b", 20, 1)])
    dataset = CocoDetection(tmp_path, manifest, transforms=None)
    assert dataset.ids == [11, 12, 21]
    assert len(dataset) == 3
