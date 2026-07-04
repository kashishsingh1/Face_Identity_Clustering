"""Cluster ArcFace embeddings into identities with no prior knowledge of how many
people are present. HDBSCAN (density-based, auto-determines cluster count, natively
supports an outlier/"noise" bucket) is the default backend; Agglomerative Clustering
with a distance threshold is offered as a fallback that can't fail via density
starvation, at the cost of losing native noise detection.
"""

from __future__ import annotations

import hdbscan
import numpy as np
from sklearn.cluster import AgglomerativeClustering

NOISE_LABEL = -1


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Precomputed pairwise cosine-distance matrix for L2-normalized embeddings.
    Explicit/auditable and trivially cheap at small-to-moderate scale (O(n^2)).
    For 100k+ images this must be replaced by an ANN-based approach -- see README's
    Scalability section.
    """
    sim = embeddings @ embeddings.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)  # guard against tiny negative floating-point noise
    return dist.astype(np.float64)  # HDBSCAN's precomputed path expects float64


class FaceClusterer:
    def __init__(
        self,
        method: str = "hdbscan",
        min_cluster_size: int = 2,
        min_samples: int = 1,
        distance_metric: str = "precomputed_cosine",
        distance_threshold: float = 0.4,
    ):
        """
        Args:
            method: "hdbscan" (default) or "agglomerative" (no-noise fallback).
            min_cluster_size: HDBSCAN's library default is 5, which would mark
                *everything* as noise on a dataset with only 2 photos per person.
                Default here is 2 -- the smallest possible cluster size -- because the
                bundled demo dataset has exactly 2 images/identity. This is
                CLI-exposed and MUST be raised (e.g. 3-5+) on denser real-world
                datasets, since 2 provides essentially no noise-filtering power at
                all and will happily "cluster" any two coincidentally-similar
                bystander faces together.
            min_samples: HDBSCAN density-strictness knob. Loosened to 1 (most
                permissive) for the same sparse-demo-dataset reason as above; raise
                for larger, denser datasets to get more conservative core-point
                estimates.
            distance_metric: "precomputed_cosine" (default) or
                "euclidean_normalized". For L2-normalized vectors,
                euclidean^2 = 2(1 - cosine_sim), so ranking of nearest neighbors is
                monotonic and the two are mathematically equivalent for HDBSCAN's
                tree structure. precomputed_cosine is the default for auditability at
                small scale; euclidean_normalized avoids materializing an O(n^2)
                matrix and should be used for larger datasets.
            distance_threshold: cosine-distance cut used only by "agglomerative".
        """
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.distance_metric = distance_metric
        self.distance_threshold = distance_threshold

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] == 0:
            return np.empty((0,), dtype=int)
        if self.method == "hdbscan":
            return self._fit_hdbscan(embeddings)
        elif self.method == "agglomerative":
            return self._fit_agglomerative(embeddings)
        raise ValueError(f"Unknown clustering method: {self.method}")

    def _fit_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] < self.min_cluster_size:
            # Not enough points to form even one cluster; everything is noise.
            return np.full(embeddings.shape[0], NOISE_LABEL, dtype=int)

        if self.distance_metric == "precomputed_cosine":
            dist = cosine_distance_matrix(embeddings)
            clusterer = hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
            )
            labels = clusterer.fit_predict(dist)
        else:  # euclidean_normalized
            clusterer = hdbscan.HDBSCAN(
                metric="euclidean",
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
            )
            labels = clusterer.fit_predict(embeddings)
        self.clusterer_ = clusterer
        return labels

    def _fit_agglomerative(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] < 2:
            return np.zeros(embeddings.shape[0], dtype=int)
        dist = cosine_distance_matrix(embeddings)
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)
        self.model_ = model
        return labels


def relabel_clusters(labels: np.ndarray) -> dict[int, str]:
    """Map raw integer labels to deterministic, human-readable names:
    person_001, person_002, ... sorted by cluster size descending, and -1 -> "unknown".
    Returns a dict {raw_label: display_name}.
    """
    unique, counts = np.unique(labels[labels != NOISE_LABEL], return_counts=True)
    order = unique[np.argsort(-counts)]
    mapping = {int(lbl): f"person_{i + 1:03d}" for i, lbl in enumerate(order)}
    mapping[NOISE_LABEL] = "unknown"
    return mapping
