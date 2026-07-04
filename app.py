"""Streamlit web UI for the face clustering pipeline.

Reuses run_pipeline() from main.py directly -- no pipeline logic is duplicated here,
this file is presentation-only. Supports uploading any number of images, persists
them to data/uploads/<run_id>/ inside the repo, shows live progress while the
pipeline runs, and renders results as cluster galleries with confidence scores plus
a downloadable CSV.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from detector import FaceDetector
from main import run_pipeline
from utils import FaceRecord, ensure_dir

UPLOADS_ROOT = Path("data/uploads")

st.set_page_config(page_title="Face Identity Clustering", layout="wide")


@st.cache_resource(show_spinner="Loading face detection & recognition model...")
def get_detector(det_size: tuple[int, int], det_thresh: float) -> FaceDetector:
    return FaceDetector(det_size=det_size, det_thresh=det_thresh)


def save_uploads(uploaded_files) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(UPLOADS_ROOT / run_id)
    for uploaded in uploaded_files:
        (run_dir / uploaded.name).write_bytes(uploaded.getvalue())
    return run_dir


def records_to_csv_bytes(records: list[FaceRecord]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["filename", "cluster_id", "confidence"])
    for record in records:
        label = record.cluster_label or "unknown"
        confidence = record.confidence if record.confidence is not None else 0.0
        writer.writerow([record.filename, label, f"{confidence:.4f}"])
    return buf.getvalue().encode("utf-8")


def render_results(records: list[FaceRecord]) -> None:
    groups: dict[str, list[FaceRecord]] = {}
    for record in records:
        groups.setdefault(record.cluster_label or "unknown", []).append(record)

    identity_labels = sorted(
        [label for label in groups if label not in ("unknown", "no_face_detected")]
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Images processed", len(records))
    col2.metric("Identities found", len(identity_labels))
    col3.metric(
        "Unknown / no-face",
        sum(len(groups.get(k, [])) for k in ("unknown", "no_face_detected")),
    )

    st.download_button(
        "Download results.csv",
        data=records_to_csv_bytes(records),
        file_name="results.csv",
        mime="text/csv",
    )

    tab_labels = identity_labels + [k for k in ("unknown", "no_face_detected") if k in groups]
    tabs = st.tabs([f"{label} ({len(groups[label])})" for label in tab_labels])

    for tab, label in zip(tabs, tab_labels):
        with tab:
            cluster_records = groups[label]
            cols = st.columns(4)
            for i, record in enumerate(cluster_records):
                with cols[i % 4]:
                    st.image(str(record.filepath), use_container_width=True)
                    conf = f"{record.confidence:.2f}" if record.confidence is not None else "n/a"
                    st.caption(f"{record.filename}\nconfidence: {conf}")


def main() -> None:
    st.title("Face Identity Clustering")
    st.write(
        "Upload any number of photos. The pipeline detects faces (RetinaFace via "
        "InsightFace), extracts ArcFace embeddings, and automatically clusters "
        "images belonging to the same person -- with no prior knowledge of how many "
        "people are in the set."
    )

    with st.sidebar:
        st.header("Pipeline settings")
        primary_face_policy = st.selectbox(
            "Primary-face selection policy",
            ["largest_area", "center_weighted", "det_score"],
            help="How to pick 'the subject' when a photo contains multiple faces (e.g. bystanders).",
        )
        cluster_method = st.selectbox("Clustering method", ["hdbscan", "agglomerative"])
        min_cluster_size = st.slider(
            "min_cluster_size (HDBSCAN)",
            2,
            10,
            2,
            help="Raise this for larger/denser datasets. 2 is appropriate only for very small demo sets.",
        )
        min_samples = st.slider("min_samples (HDBSCAN)", 1, 10, 1)
        det_thresh = st.slider("Detection confidence threshold", 0.1, 0.9, 0.5)
        distance_threshold = st.slider(
            "distance_threshold (Agglomerative)", 0.1, 1.0, 0.4, 0.05
        )
        unknown_confidence_threshold = st.slider(
            "Unknown confidence threshold (Agglomerative)", 0.0, 1.0, 0.5, 0.05
        )

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Run Clustering", type="primary"):
        run_dir = save_uploads(uploaded_files)
        image_paths = sorted(run_dir.iterdir(), key=lambda p: p.name)

        detector = get_detector((640, 640), det_thresh)

        with st.status("Running pipeline...", expanded=True) as status:
            progress_bar = st.progress(0.0)

            def progress(stage: str, current: int, total: int) -> None:
                status.write(f"{stage}: {current}/{total}")
                if total > 0:
                    progress_bar.progress(min(1.0, current / total))

            records = run_pipeline(
                image_paths,
                det_thresh=det_thresh,
                primary_face_policy=primary_face_policy,
                cluster_method=cluster_method,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                distance_threshold=distance_threshold,
                unknown_confidence_threshold=unknown_confidence_threshold,
                detector=detector,
                progress=progress,
            )
            status.update(label="Pipeline complete", state="complete")

        st.session_state["records"] = records

    if "records" in st.session_state:
        render_results(st.session_state["records"])


if __name__ == "__main__":
    main()
