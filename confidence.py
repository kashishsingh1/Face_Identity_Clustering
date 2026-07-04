"""Per-image confidence scoring: cosine similarity between an image's embedding and
its assigned cluster's centroid. Noise/unclustered points get their best similarity to
any existing centroid, so every image always receives a real number (never blank/NaN).
"""

from __future__ import annotations

import numpy as np

from clustering import NOISE_LABEL


class ConfidenceScorer:
    def compute_centroids(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> dict[int, np.ndarray]:
        centroids: dict[int, np.ndarray] = {}
        for lbl in set(labels.tolist()):
            if lbl == NOISE_LABEL:
                continue
            members = embeddings[labels == lbl]
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[int(lbl)] = centroid / norm if norm > 1e-12 else centroid
        return centroids

    def score(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        centroids = self.compute_centroids(embeddings, labels)
        raw_cos_sim = np.zeros(len(labels), dtype=np.float32)

        for i, lbl in enumerate(labels):
            lbl = int(lbl)
            if lbl != NOISE_LABEL and lbl in centroids:
                raw_cos_sim[i] = float(np.dot(embeddings[i], centroids[lbl]))
            elif centroids:
                sims = [float(np.dot(embeddings[i], c)) for c in centroids.values()]
                raw_cos_sim[i] = max(sims)
            else:
                raw_cos_sim[i] = -1.0  # no clusters exist at all -> rescales to 0.0

        # Rescale cosine similarity [-1, 1] -> [0, 1] for a friendlier "confidence"
        # semantic. This is a linear rescale, not a probability estimate.
        confidence = np.clip((raw_cos_sim + 1.0) / 2.0, 0.0, 1.0)
        return confidence.astype(np.float32)
