"""CLI entrypoint for the face clustering pipeline.

The core orchestration lives in run_pipeline() so both this CLI and app.py (the
Streamlit UI) share exactly one implementation of the detect -> embed -> cluster ->
score -> write-outputs flow.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np

from clustering import NOISE_LABEL, FaceClusterer, relabel_clusters
from confidence import ConfidenceScorer
from detector import FaceDetector, select_primary_face
from embeddings import EmbeddingExtractor
from utils import (
    NO_FACE_SENTINEL,
    FaceRecord,
    copy_to_output,
    ensure_dir,
    list_images,
    load_image_bgr,
    set_seed,
    setup_logger,
)
from visualization import ClusterVisualizer

ProgressCallback = Callable[[str, int, int], None]


def run_pipeline(
    image_paths: list[Path],
    *,
    det_size: tuple[int, int] = (640, 640),
    det_thresh: float = 0.5,
    primary_face_policy: str = "largest_area",
    cluster_method: str = "hdbscan",
    min_cluster_size: int = 2,
    min_samples: int = 1,
    distance_metric: str = "precomputed_cosine",
    distance_threshold: float = 0.4,
    unknown_confidence_threshold: float = 0.5,
    detector: FaceDetector | None = None,
    progress: ProgressCallback | None = None,
) -> list[FaceRecord]:
    """Run the full detect -> select-primary-face -> embed -> cluster -> score flow
    over a list of image paths, returning one FaceRecord per input path.

    A progress callback, if given, is invoked as progress(stage_name, current, total)
    so a caller (e.g. a Streamlit UI) can render live progress without this function
    knowing anything about the UI layer.
    """

    def report(stage: str, current: int, total: int) -> None:
        if progress is not None:
            progress(stage, current, total)

    detector = detector or FaceDetector(det_size=det_size, det_thresh=det_thresh)
    extractor = EmbeddingExtractor()

    records: list[FaceRecord] = []
    embeddings: list[np.ndarray] = []
    embedded_indices: list[int] = []

    total = len(image_paths)
    for i, path in enumerate(image_paths):
        report("detecting", i + 1, total)
        record = FaceRecord(filename=path.name, filepath=path)
        try:
            image = load_image_bgr(path)
        except Exception as exc:  # corrupt/unreadable file: skip, don't crash the run
            record.error = str(exc)
            records.append(record)
            continue

        faces = detector.detect(image)
        record.num_faces_detected = len(faces)
        primary = select_primary_face(faces, policy=primary_face_policy, image_shape=image.shape)

        if primary is None:
            record.cluster_id = NO_FACE_SENTINEL
            record.cluster_label = "no_face_detected"
            record.confidence = 0.0
            records.append(record)
            continue

        record.bbox = primary.bbox
        record.det_score = float(primary.det_score)
        record.kps = primary.kps
        record.embedding = extractor.extract(primary)
        embedded_indices.append(len(records))
        embeddings.append(record.embedding)
        records.append(record)

    if embeddings:
        report("clustering", 0, 1)
        embedding_matrix = np.stack(embeddings)
        clusterer = FaceClusterer(
            method=cluster_method,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            distance_metric=distance_metric,
            distance_threshold=distance_threshold,
        )
        raw_labels = clusterer.fit(embedding_matrix)

        report("scoring", 0, 1)
        confidences = ConfidenceScorer().score(embedding_matrix, raw_labels)

        if cluster_method == "agglomerative":
            # Agglomerative has no native noise concept: assign every point to some
            # cluster, then reroute low-confidence ones to "unknown" post-hoc so the
            # unknown-bucket UX is preserved.
            raw_labels = raw_labels.copy()
            low_conf_mask = confidences < unknown_confidence_threshold
            raw_labels[low_conf_mask] = NOISE_LABEL

        label_names = relabel_clusters(raw_labels)

        for idx, raw_label, confidence in zip(embedded_indices, raw_labels, confidences):
            records[idx].cluster_id = int(raw_label)
            records[idx].cluster_label = label_names[int(raw_label)]
            records[idx].confidence = float(confidence)

        report("clustering", 1, 1)

    return records


def write_results(records: list[FaceRecord], output_dir: Path) -> Path:
    output_dir = ensure_dir(output_dir)
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "cluster_id", "confidence"])
        for record in records:
            label = record.cluster_label or "unknown"
            confidence = record.confidence if record.confidence is not None else 0.0
            writer.writerow([record.filename, label, f"{confidence:.4f}"])

    for record in records:
        label = record.cluster_label or "unknown"
        copy_to_output(record.filepath, output_dir / label)

    return csv_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster face images by identity.")
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--det-size", type=str, default="640,640")
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument(
        "--primary-face-policy",
        choices=["largest_area", "center_weighted", "det_score"],
        default="largest_area",
    )
    parser.add_argument("--cluster-method", choices=["hdbscan", "agglomerative"], default="hdbscan")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument(
        "--distance-metric",
        choices=["precomputed_cosine", "euclidean_normalized"],
        default="precomputed_cosine",
    )
    parser.add_argument("--distance-threshold", type=float, default=0.4)
    parser.add_argument("--unknown-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run diagnostic evaluation against filename-derived ground truth.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logger = setup_logger(verbose=args.verbose)
    set_seed(args.seed)

    det_w, det_h = (int(v) for v in args.det_size.split(","))

    image_paths = list_images(args.input)
    if not image_paths:
        logger.warning(f"No images found in {args.input}")
        return
    logger.info(f"Found {len(image_paths)} images in {args.input}")

    def progress(stage: str, current: int, total: int) -> None:
        logger.info(f"[{stage}] {current}/{total}")

    records = run_pipeline(
        image_paths,
        det_size=(det_w, det_h),
        det_thresh=args.det_thresh,
        primary_face_policy=args.primary_face_policy,
        cluster_method=args.cluster_method,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        distance_metric=args.distance_metric,
        distance_threshold=args.distance_threshold,
        unknown_confidence_threshold=args.unknown_confidence_threshold,
        progress=progress,
    )

    csv_path = write_results(records, args.output)
    logger.info(f"Wrote {csv_path}")

    cluster_labels = {r.cluster_label for r in records if r.cluster_label not in (None, "unknown", "no_face_detected")}
    unknown_count = sum(1 for r in records if r.cluster_label == "unknown")
    no_face_count = sum(1 for r in records if r.cluster_label == "no_face_detected")
    logger.info(
        f"Summary: {len(records)} images processed, {len(cluster_labels)} identities found, "
        f"{unknown_count} unknown, {no_face_count} no-face-detected"
    )

    if not args.skip_visualization:
        ClusterVisualizer().render_all(records, args.output / "visualizations")
        logger.info(f"Wrote visualizations to {args.output / 'visualizations'}")

    if args.eval:
        from eval import evaluate

        metrics = evaluate(records)
        logger.info(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
