"""ArcFace embedding extraction.

Note: alignment already happened upstream. insightface's FaceAnalysis.get() performs a
5-point-landmark affine warp internally before running the ArcFace (w600k_r50)
recognition model, so `face.normed_embedding` is already an aligned, L2-normalized
512-d embedding. This module exists to make that step an explicit, testable,
swappable unit rather than something implicitly buried inside detector.py, and to
defensively guarantee normalization regardless of upstream changes.
"""

from __future__ import annotations

import numpy as np


class EmbeddingExtractor:
    def __init__(self, l2_normalize: bool = True):
        self.l2_normalize = l2_normalize

    def extract(self, face) -> np.ndarray:
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = face.embedding
        embedding = np.asarray(embedding, dtype=np.float32)
        if self.l2_normalize:
            norm = np.linalg.norm(embedding)
            if norm > 1e-12:
                embedding = embedding / norm
        return embedding

    def extract_batch(self, faces: list) -> np.ndarray:
        if not faces:
            return np.empty((0, 512), dtype=np.float32)
        return np.stack([self.extract(f) for f in faces])
