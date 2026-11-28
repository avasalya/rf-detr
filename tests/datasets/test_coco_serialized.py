# ------------------------------------------------------------------------
# COMET3D: tests for serialized train annotations
# Copyright (c) 2026 Ashesh Vasalya (avasalya, asheshvasalya@gmail.com). All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
"""``CocoDetection(serialize_annotations=True)`` returns the same samples while its workers share the annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.datasets.coco import CocoDetection, build_coco, build_roboflow_from_coco

KEYPOINT_FIELDS = {"keypoints": 17, "foot_kpts": 6, "face_kpts": 68, "lefthand_kpts": 21, "righthand_kpts": 21}


def write_keypoint_coco(directory: Path) -> Path:
    """Write a 3-image whole-body COCO split (image 2 has no annotations, image 3 a crowd one); return its json.

    Examples:
        >>> import tempfile
        >>> ann_file = write_keypoint_coco(Path(tempfile.mkdtemp()))
        >>> [len(json.loads(ann_file.read_text())[key]) for key in ("images", "annotations")]
        [3, 4]
    """
    images = []
    for image_id, (width, height) in enumerate([(32, 24), (16, 16), (24, 32)], start=1):
        Image.new("RGB", (width, height), color=(image_id * 40, 0, 0)).save(directory / f"{image_id}.jpg")
        images.append({"id": image_id, "file_name": f"{image_id}.jpg", "width": width, "height": height})

    def annotation(annotation_id: int, image_id: int, offset: float, iscrowd: int = 0) -> dict[str, Any]:
        fields = {
            name: [value for k in range(count) for value in (offset + k * 0.1, offset + k * 0.2, 2.0)]
            for name, count in KEYPOINT_FIELDS.items()
        }
        return {"id": annotation_id, "image_id": image_id, "category_id": 1, "bbox": [offset, offset, 8.5, 6.25],
                "area": 53.1, "iscrowd": iscrowd, "num_keypoints": 17, **fields}

    annotations = [annotation(1, 1, 1.0), annotation(2, 1, 3.5), annotation(3, 3, 2.0), annotation(4, 3, 0.5, 1)]
    categories = [{"id": 1, "name": "person", "keypoints": [str(k) for k in range(17)]}]
    ann_file = directory / "annotations.json"
    ann_file.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return ann_file


def build_pair(directory: Path) -> tuple[CocoDetection, CocoDetection]:
    """Build the split twice: the plain dataset and the serialized one.

    Examples:
        >>> import tempfile
        >>> plain, serialized = build_pair(Path(tempfile.mkdtemp()))  # doctest: +ELLIPSIS
        loading annotations into memory...
        >>> len(plain) == len(serialized) == 3
        True
    """
    ann_file = write_keypoint_coco(directory)
    kwargs = dict(transforms=None, include_keypoints=True, num_keypoints_per_class=[133], remap_category_ids=True)
    return CocoDetection(directory, ann_file, **kwargs), CocoDetection(
        directory, ann_file, serialize_annotations=True, **kwargs
    )


def assert_same_sample(expected: tuple[Any, dict[str, Any]], actual: tuple[Any, dict[str, Any]]) -> None:
    """Images are pixel-identical and target tensors equal key by key.

    Examples:
        >>> image = Image.new("RGB", (2, 2))
        >>> assert_same_sample((image, {"boxes": torch.zeros(1, 4)}), (image, {"boxes": torch.zeros(1, 4)}))
    """
    assert np.array_equal(np.asarray(expected[0]), np.asarray(actual[0]))
    assert expected[1].keys() == actual[1].keys()
    for key, value in expected[1].items():
        assert torch.equal(value, actual[1][key]), key


class TestSerializedAnnotations:
    def test_samples_identical_to_coco_lookup(self, tmp_path: Path) -> None:
        plain, serialized = build_pair(tmp_path)
        for index in range(len(plain)):
            assert_same_sample(plain[index], serialized[index])

    def test_samples_identical_through_worker_processes(self, tmp_path: Path) -> None:
        plain, serialized = build_pair(tmp_path)
        loader = torch.utils.data.DataLoader(serialized, batch_size=None, num_workers=2)
        for index, sample in enumerate(loader):
            assert_same_sample(plain[index], sample)

    def test_annotation_index_emptied_metadata_kept(self, tmp_path: Path) -> None:
        plain, serialized = build_pair(tmp_path)
        coco = serialized.coco
        assert (coco.anns, coco.dataset["annotations"], coco.getAnnIds(imgIds=serialized.ids)) == ({}, [], [])
        assert (coco.imgs, coco.cats, coco.label2cat) == (plain.coco.imgs, plain.coco.cats, plain.coco.label2cat)

    def test_mutating_a_sample_does_not_change_the_next_read(self, tmp_path: Path) -> None:
        _, serialized = build_pair(tmp_path)
        serialized._records[0][1][0]["bbox"][0] = 999.0
        assert serialized._records[0][1][0]["bbox"][0] == 1.0


def call_builder(builder: Any, image_set: str, compute_train_metrics: bool) -> bool:
    """Run ``builder`` with the dataset class mocked; return the ``serialize_annotations`` it passed.

    Examples:
        >>> call_builder(build_coco, "val", False)
        False
    """
    args = _namespace_from_configs(RFDETRSmallConfig(), TrainConfig(dataset_dir="/fake/dataset"))
    args.compute_train_metrics = compute_train_metrics
    with (
        patch("rfdetr.datasets.coco.CocoDetection") as dataset,
        patch("rfdetr.datasets.coco.make_coco_transforms_square_div_64", return_value=MagicMock()),
        patch("rfdetr.datasets.coco.make_coco_transforms", return_value=MagicMock()),
        patch("rfdetr.datasets.coco.Path") as path,
        patch("rfdetr.datasets.coco.logger"),
    ):
        path.return_value.exists.return_value = True
        builder(image_set, args, resolution=512)
    return dataset.call_args.kwargs["serialize_annotations"]


class TestBuildersSerializeOnlyUnevaluatedTrainSplit:
    @pytest.mark.parametrize("builder", [build_coco, build_roboflow_from_coco])
    @pytest.mark.parametrize(
        "image_set,compute_train_metrics,expected",
        [("train", False, True), ("train", True, False), ("val", False, False), ("test", False, False)],
    )
    def test_serialize_flag(self, builder: Any, image_set: str, compute_train_metrics: bool, expected: bool) -> None:
        assert call_builder(builder, image_set, compute_train_metrics) is expected
