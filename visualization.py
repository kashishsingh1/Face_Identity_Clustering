"""Render per-cluster montage images for demo/README purposes."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless-safe: no display required
import matplotlib.pyplot as plt

from utils import FaceRecord


class ClusterVisualizer:
    def __init__(self, thumb_size: tuple[int, int] = (160, 160), max_cols: int = 4):
        self.thumb_size = thumb_size
        self.max_cols = max_cols

    def _load_thumb(self, record: FaceRecord, draw_bbox: bool = True):
        image = cv2.imread(str(record.filepath))
        if image is None:
            return None
        if draw_bbox and record.bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in record.bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        image = cv2.resize(image, self.thumb_size)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def render_cluster_grid(
        self, records: list[FaceRecord], cluster_label: str, out_path: Path
    ) -> Path:
        n = len(records)
        cols = min(self.max_cols, max(1, n))
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
        axes = [axes] if n == 1 else axes.flatten()

        for ax, record in zip(axes, records):
            thumb = self._load_thumb(record)
            if thumb is not None:
                ax.imshow(thumb)
            conf = f"{record.confidence:.2f}" if record.confidence is not None else "n/a"
            ax.set_title(f"{record.filename}\nconf={conf}", fontsize=8)
            ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(f"{cluster_label} (n={n})")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    def render_all(self, records: list[FaceRecord], out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        groups: dict[str, list[FaceRecord]] = {}
        for record in records:
            groups.setdefault(record.cluster_label or "unknown", []).append(record)

        rendered = []
        for label, group_records in sorted(groups.items()):
            path = out_dir / f"{label}.png"
            rendered.append(self.render_cluster_grid(group_records, label, path))
        return rendered
