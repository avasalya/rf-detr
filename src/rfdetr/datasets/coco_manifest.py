# ------------------------------------------------------------------------
# COMET3D: per-source COCO annotation files read as one dataset (annotations/train/manifest.json)
# Copyright (c) 2026 Ashesh Vasalya (avasalya, asheshvasalya@gmail.com). All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
"""One COCO index from per-source annotation files listed in a manifest.

A large merged dataset is stored as one COCO json per source plus ``manifest.json``, so updating one source rewrites
only its file. Every source owns a fixed id range, so the files concatenate without renumbering.

``manifest.json``::

    {"format": 1, "id_block": 10000000, "categories": [...],
     "sources": [{"name": "coco", "file": "coco.json", "images": 118287, "annotations": 262461}, ...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO

MANIFEST_NAME = "manifest.json"


def is_manifest(path: str | Path) -> bool:
    """Whether ``path`` names a per-source annotation manifest.

    Args:
        path: Annotation file path.

    Returns:
        ``True`` for a file called ``manifest.json``.

    Examples:
        >>> is_manifest("annotations/train/manifest.json"), is_manifest("annotations/person_keypoints_train2017.json")
        (True, False)
    """
    return Path(path).name == MANIFEST_NAME


def load_manifest_coco(manifest_path: str | Path) -> COCO:
    """Build one pycocotools ``COCO`` index from the per-source files of a manifest.

    Args:
        manifest_path: Path to ``manifest.json``; source files are resolved relative to its directory.

    Returns:
        A ``COCO`` object whose ``dataset`` holds the images and annotations of every listed source, in manifest
        order, with the manifest's categories.

    Raises:
        ValueError: If a source file's image / annotation count differs from the manifest (stale or partial write) or
            an image / annotation id appears twice.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for entry in manifest["sources"]:
        with (manifest_path.parent / entry["file"]).open() as f:
            data = json.load(f)
        if (len(data["images"]), len(data["annotations"])) != (entry["images"], entry["annotations"]):
            raise ValueError(
                f"{entry['file']}: {len(data['images'])} images, {len(data['annotations'])} annotations; "
                f"manifest says {entry['images']}, {entry['annotations']}"
            )
        images.extend(data["images"])
        annotations.extend(data["annotations"])
        del data
    for kind, records in (("image", images), ("annotation", annotations)):
        if len({record["id"] for record in records}) != len(records):
            raise ValueError(f"{manifest_path}: duplicate {kind} ids across source files")
    coco = COCO()
    coco.dataset = {
        "info": {"description": f"manifest: {', '.join(e['name'] for e in manifest['sources'])}"},
        "images": images,
        "annotations": annotations,
        "categories": manifest["categories"],
    }
    coco.createIndex()
    return coco
