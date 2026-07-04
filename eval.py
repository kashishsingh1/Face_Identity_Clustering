"""Diagnostic evaluation of clustering quality against ground-truth identities parsed
from filenames (e.g. 'person_01_0.jpg' -> 'person_01').

This is diagnostic-only: it exists to validate pipeline correctness for the demo,
and is never imported by the production detector/embeddings/clustering/confidence
modules. main.py's --eval flag calls evaluate() directly on in-memory records; this
script can also be run standalone against a previously written results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)

from utils import FaceRecord, parse_ground_truth_from_filename


def _purity(y_true: list[str], y_pred: list[str]) -> float:
    from collections import Counter, defaultdict

    clusters: dict[str, list[str]] = defaultdict(list)
    for true_label, pred_label in zip(y_true, y_pred):
        clusters[pred_label].append(true_label)

    total_correct = sum(Counter(members).most_common(1)[0][1] for members in clusters.values())
    return total_correct / len(y_true) if y_true else 0.0


def _per_cluster_breakdown(y_true: list[str], y_pred: list[str]) -> dict[str, dict]:
    from collections import Counter, defaultdict

    clusters: dict[str, list[str]] = defaultdict(list)
    for true_label, pred_label in zip(y_true, y_pred):
        clusters[pred_label].append(true_label)

    breakdown = {}
    for pred_label, members in clusters.items():
        dominant, count = Counter(members).most_common(1)[0]
        breakdown[pred_label] = {
            "size": len(members),
            "dominant_true_identity": dominant,
            "cluster_purity": count / len(members),
        }
    return breakdown


def evaluate(records: list[FaceRecord]) -> dict:
    """Compute ARI, homogeneity/completeness/V-measure, and purity against
    filename-derived ground truth. Records whose filename doesn't match the
    'person_XX_*' pattern are dropped from evaluation (with a count reported)."""
    y_true: list[str] = []
    y_pred: list[str] = []
    skipped = 0

    for record in records:
        truth = parse_ground_truth_from_filename(record.filename)
        if truth is None:
            skipped += 1
            continue
        y_true.append(truth)
        y_pred.append(record.cluster_label or "unknown")

    if not y_true:
        return {"error": "no ground-truth-parseable filenames found", "skipped": skipped}

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)

    return {
        "num_evaluated": len(y_true),
        "num_skipped": skipped,
        "adjusted_rand_index": adjusted_rand_score(y_true, y_pred),
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "purity": _purity(y_true, y_pred),
        "per_cluster": _per_cluster_breakdown(y_true, y_pred),
    }


def evaluate_from_csv(csv_path: Path) -> dict:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            record = FaceRecord(filename=row["filename"], filepath=Path(row["filename"]))
            record.cluster_label = row["cluster_id"]
            record.confidence = float(row["confidence"])
            records.append(record)
    return evaluate(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a results.csv against filename-derived ground truth.")
    parser.add_argument("--results", type=Path, default=Path("output/results.csv"))
    args = parser.parse_args()

    metrics = evaluate_from_csv(args.results)
    print(json.dumps(metrics, indent=2))
