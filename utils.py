"""Shared low-level helpers: data model, image I/O, logging. No ML logic lives here."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Sentinel cluster ids used before a human-readable label is assigned.
NO_FACE_SENTINEL = -2
NOISE_SENTINEL = -1


class ImageLoadError(Exception):
    """Raised when an image file exists but cannot be decoded."""


@dataclass
class FaceRecord:
    """One row of pipeline state per input image file."""

    filename: str
    filepath: Path
    bbox: np.ndarray | None = None
    det_score: float | None = None
    kps: np.ndarray | None = None
    embedding: np.ndarray | None = None
    cluster_id: int | None = None
    cluster_label: str | None = None
    confidence: float | None = None
    num_faces_detected: int = 0
    error: str | None = None


def list_images(input_dir: Path) -> list[Path]:
    """Return a sorted list of image paths in input_dir (non-recursive)."""
    input_dir = Path(input_dir)
    paths = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name)


def load_image_bgr(path: Path) -> np.ndarray:
    """Load an image as a BGR numpy array (OpenCV convention). Raises ImageLoadError on failure."""
    data = cv2.imread(str(path))
    if data is None:
        raise ImageLoadError(f"Could not decode image: {path}")
    return data


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_to_output(src: Path, dest_dir: Path) -> Path:
    """Copy a file into dest_dir, preserving metadata. Uses copy2 (not symlinks) for
    Windows portability -- symlinks require elevated privileges/dev mode on Windows."""
    dest_dir = ensure_dir(dest_dir)
    dest = dest_dir / Path(src).name
    shutil.copy2(src, dest)
    return dest


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def setup_logger(name: str = "face_pipeline", verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger


_GROUND_TRUTH_PATTERN = re.compile(r"^(person_\d+)_")


def parse_ground_truth_from_filename(filename: str) -> str | None:
    """Extract a ground-truth identity label from a filename like 'person_01_0.jpg' -> 'person_01'.

    DIAGNOSTIC USE ONLY: this must be called exclusively from eval.py / main.py's --eval
    path, never from the production detect->embed->cluster->confidence pipeline. The
    pipeline must never use filenames as a shortcut for identity.
    """
    match = _GROUND_TRUTH_PATTERN.match(filename)
    return match.group(1) if match else None
