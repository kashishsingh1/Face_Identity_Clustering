# Face Identity Clustering

Automatically group photos of the same person from an unorganized image folder — no
prior knowledge of how many people are in the set, no labels, no manual sorting.
Every image gets a **confidence score** for its identity match, and low-confidence /
undetectable faces are routed to an explicit `unknown` bucket rather than silently
mis-clustered.

Given a folder of photos, the system:
1. Detects every face in each photo (RetinaFace via InsightFace) and picks the one
   representing the photo's subject (handling background bystanders).
2. Extracts an aligned, L2-normalized ArcFace embedding for that face.
3. Clusters all embeddings with HDBSCAN — automatically discovering the number of
   identities, with no `k` specified up front.
4. Scores each image's confidence as its cosine similarity to its cluster's centroid.
5. Writes `person_001/`, `person_002/`, ... and `unknown/` folders, plus a
   `filename,cluster_id,confidence` CSV.

A CLI (`main.py`) and a Streamlit web app (`app.py`) both drive the same pipeline
code — see [Running inference](#running-inference).

## Example output

Run against the bundled 6-image reference set (3 people, 2 photos each, taken at a
bowling alley with dim colored lighting, motion blur, and 2–4 other bystanders
visible in every frame):

```
Summary: 6 images processed, 3 identities found, 0 unknown, 0 no-face-detected
Evaluation metrics: {
  'adjusted_rand_index': 1.0, 'homogeneity': 1.0, 'completeness': 1.0,
  'v_measure': 1.0, 'purity': 1.0
}
```

`output/results.csv`:

| filename          | cluster_id | confidence |
|-------------------|------------|------------|
| person_01_0.jpg   | person_002 | 0.9590     |
| person_01_1.jpg   | person_002 | 0.9590     |
| person_02_0.jpg   | person_003 | 0.9414     |
| person_02_1.jpg   | person_003 | 0.9414     |
| person_03_0.jpg   | person_001 | 0.9641     |
| person_03_1.jpg   | person_001 | 0.9641     |

After running, `output/visualizations/person_002.png` (generated locally, not
committed — see [privacy note](#dataset-setup)) shows a montage with a green box
drawn around the *primary face the pipeline selected* for each image, confirming it
correctly ignored the bystanders visible behind the subject in both frames.

Despite the low, colored lighting and busy backgrounds, all 3 identities were
recovered perfectly (ARI = purity = 1.0) with confidence scores around 0.94–0.96.

## Installation

Requires Python 3.14 (verified working; earlier 3.10–3.13 should also work as all
pinned dependencies ship wheels for them).

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

On first run, InsightFace downloads its `buffalo_l` model pack (~280 MB) from a
GitHub release mirror to `%USERPROFILE%\.insightface\models\buffalo_l\`. This
requires internet access once; after that, it's cached locally and every subsequent
run is fully offline. To pre-seed an offline machine, copy that folder over from a
machine that has already run the pipeline once.

## Dependencies

All pinned in `requirements.txt`: `insightface`, `onnxruntime` (CPU — no CUDA
required), `numpy`, `opencv-python`, `scikit-learn`, `scipy`, `hdbscan`,
`matplotlib`, `pandas`, `Pillow`, `tqdm`, `streamlit`, plus InsightFace's own
transitive dependencies (`scikit-image`, `easydict`, `Cython`, `requests`).

## Dataset setup

Drop any folder of images into `input/` (jpg/jpeg/png/bmp/webp). The bundled
reference set (`input/.gitkeep` only — the actual sample images are **not**
committed, since they're identifiable photos of real people) can be restored by
copying images into `input/` yourself, or by uploading them through the Streamlit
app instead.

**Important:** filenames are never read by the production pipeline. The only place
a filename pattern (`person_XX_*`) is parsed is `eval.py`, purely as a diagnostic
against known ground truth — the detection/embedding/clustering/confidence code
never sees or depends on filenames.

## Running inference

### CLI

```bash
python main.py --input input/ --output output/ --eval
```

Small demo-sized dataset (2 images/person) — default settings already handle this:

```bash
python main.py --input input/ --output output/ --min-cluster-size 2 --eval
```

Larger, denser dataset — raise `min_cluster_size` so coincidental 2-image overlaps
between bystanders don't form spurious clusters:

```bash
python main.py --input input/ --output output/ --min-cluster-size 5
```

Agglomerative fallback (if HDBSCAN produces excessive noise):

```bash
python main.py --input input/ --output output/ --cluster-method agglomerative --distance-threshold 0.4
```

All flags:

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `input/` | Folder of images |
| `--output` | `output/` | Output folder |
| `--det-size` | `640,640` | Detector input resolution |
| `--det-thresh` | `0.5` | Face detection confidence threshold |
| `--primary-face-policy` | `largest_area` | `largest_area` \| `center_weighted` \| `det_score` |
| `--cluster-method` | `hdbscan` | `hdbscan` \| `agglomerative` |
| `--min-cluster-size` | `2` | HDBSCAN min cluster size (see [Design decisions](#design-decisions)) |
| `--min-samples` | `1` | HDBSCAN density strictness |
| `--distance-metric` | `precomputed_cosine` | `precomputed_cosine` \| `euclidean_normalized` |
| `--distance-threshold` | `0.4` | Agglomerative cosine-distance cut |
| `--unknown-confidence-threshold` | `0.5` | Agglomerative-only: reroute low-confidence points to `unknown` |
| `--skip-visualization` | off | Skip generating cluster montage PNGs |
| `--eval` | off | Run diagnostic metrics against filename-derived ground truth |
| `--verbose` / `-v` | off | Verbose logging |
| `--seed` | `42` | RNG seed |

### Streamlit web app (recommended for demos)

```bash
streamlit run app.py
```

Upload any number of images (no cap), tune parameters in the sidebar, click **Run
Clustering**, and browse results as tabbed cluster galleries with confidence badges,
summary metric cards, and a CSV download button. Uploaded images are saved to
`data/uploads/<timestamp>/` inside the repo (also gitignored) so re-runs against the
same set don't require re-uploading from scratch.

### Standalone evaluation

```bash
python eval.py --results output/results.csv
```

## Design decisions

### Why InsightFace / ArcFace?

`insightface.app.FaceAnalysis` bundles a RetinaFace-based detector, 5-point-landmark
alignment, and the ArcFace `w600k_r50` recognition model behind one ONNX Runtime CPU
pipeline. Wiring separate detector/aligner/recognizer repos together would add
integration surface for no accuracy benefit here — InsightFace is a single,
well-maintained dependency that runs entirely on CPU (no CUDA needed for a
demo/laptop), and ArcFace's additive angular margin loss produces embeddings with
strong inter-class separation and intra-class compactness, which is exactly what a
downstream clustering step needs. Alignment happens automatically inside
`FaceAnalysis.get()` before the embedding is computed (see `embeddings.py`'s
docstring) — there's no separate visible "align" step because it's not a separate
model call, but it is not skipped.

### Handling multiple faces per photo (the bystander problem)

The bundled reference dataset is not cropped face photos — it's full-body bowling
alley snapshots with 2–4 other people visible in the background of every frame.
`detector.py`'s `select_primary_face()` picks the largest detected bounding box as
"the subject" by default (`--primary-face-policy largest_area`), since the
foreground subject reliably fills more of the frame than smaller, often
partially-occluded bystanders. Two documented fallbacks exist for datasets where
this assumption breaks: `center_weighted` (rewards large *and* centered faces) and
`det_score` (pure detector-confidence argmax). None of these can fully solve the case
where a bystander happens to be closer to the camera than the actual subject — a
production fix would need multi-frame subject tracking or a manual selection UI,
which is out of scope for single stills.

### Why HDBSCAN, and why `min_cluster_size=2`?

HDBSCAN was chosen over k-means (which requires a known `k`, defeating the "no prior
knowledge of identity count" requirement) and over plain DBSCAN (which uses a single
global density threshold — HDBSCAN instead builds a cluster hierarchy and extracts
stable clusters at varying densities, which matters when some identities have many
photos and others have few). It also gives a native "noise" label (`-1`) for
outliers, which maps directly onto the required `unknown/` bucket.

**`min_cluster_size=2`** (not HDBSCAN's library default of 5) is the single most
important non-default parameter in this repo, and it is dataset-driven: the bundled
reference set has exactly 2 photos per identity. At the default of 5, HDBSCAN would
mark *every* point as noise — a silent, plausible-looking failure (the run would
"succeed" with 0 identities found). **If you run this on a denser real-world
dataset, raise `--min-cluster-size` to at least the smallest expected number of
photos per identity** — a value of 2 provides essentially zero noise-filtering
power and will happily "cluster" two coincidentally-similar bystander faces
together. A reasonable default for realistic datasets (dozens+ images/person) is
3–5. `--min-samples` follows the same logic (loosened to `1` here for the same
reason).

### Cosine distance: precomputed vs. the euclidean-on-normalized trick

For L2-normalized embeddings, `euclidean² = 2(1 − cosine_sim)`, so ranking by
Euclidean distance and by cosine distance is monotonically identical — HDBSCAN's
`metric="euclidean"` on normalized vectors is mathematically equivalent to using
cosine distance directly, and is the standard trick used to avoid materializing an
O(n²) distance matrix. This repo defaults to explicitly computing the
**precomputed cosine-distance matrix** (`--distance-metric precomputed_cosine`)
instead, because at the dataset sizes this tool targets (hundreds to low
thousands of images) the O(n²) cost is trivial, and having an explicit, inspectable
distance matrix makes debugging and the eval/demo narrative more transparent. The
`euclidean_normalized` alternative is exposed via CLI/UI for anyone who wants the
lower-memory path immediately.

### Confidence scoring

Confidence = cosine similarity between an image's embedding and its cluster's
centroid (mean of member embeddings, re-normalized), linearly rescaled from
`[-1, 1]` to `[0, 1]`. This is not a calibrated probability — it's a relative
similarity score — but in practice same-identity ArcFace cosine similarities are
almost always well above the midpoint (0.3–0.9+), so the rescale rarely matters
below 0.5. Noise/unclustered points are scored by their **best** similarity to any
existing cluster's centroid, so every row in the CSV always has a real number
(never blank), even for images that ended up in `unknown/`.

### Lighting, pose, occlusion, blur

- **Colored/dim lighting**: ArcFace's training data includes substantial
  illumination augmentation, so moderate color casts (the reference set's
  purple/pink bowling-alley lighting) are tolerated well, as shown by the 0.94–0.96
  confidence scores achieved here. Extreme under/over-exposure still degrades
  detection confidence before it ever reaches the embedding step — `--det-thresh`
  is exposed to loosen detection sensitivity if faces are being missed.
- **Profile faces**: RetinaFace's landmark regression degrades on extreme profile
  views (beyond roughly ±60–70°), which cascades into worse alignment and a noisier
  embedding. There is no dedicated pose-correction step in this pipeline; a
  production system at scale would want a pose-quality filter that flags/discounts
  extreme-profile detections rather than embedding them at full confidence.
- **Occlusion**: partial occlusion (hands, other people's heads, cropped frame
  edges) reduces the effective information in the aligned crop, generally lowering
  the embedding's discriminative power and thus confidence scores rather than
  causing outright failures — this is a soft degradation, not a hard failure mode.
- **Motion blur**: degrades landmark precision (misalignment) and to a lesser
  extent embedding quality; `--det-size` can be increased to give the detector more
  resolution to work with on blurry/small faces, at a latency cost.

### Scalability to 100,000+ images

The precomputed cosine-distance matrix is O(n²) in both memory and compute — a
100k-image dataset would need a dense ~80 GB float64 matrix, which is infeasible on
a single machine. At that scale:
- Switch clustering to `--distance-metric euclidean_normalized`, which lets
  HDBSCAN use its built-in tree-based nearest-neighbor structures (e.g. Boruvka/KD-tree)
  instead of an explicit matrix.
- For further scale, move to an ANN-index-based approach: build a k-NN graph with
  FAISS (or similar) over the embeddings, then run HDBSCAN's graph-based / precomputed
  sparse-graph clustering path on that approximate neighborhood graph rather than
  the full pairwise matrix.
- Batch face detection/embedding on GPU (`ctx_id=0` instead of `-1`) — this
  pipeline defaults to CPU since no CUDA device was available in development, but
  InsightFace/ONNX Runtime support GPU execution providers directly.
- For incremental datasets (new images arriving over time), maintain a persistent
  FAISS index of embeddings + centroids rather than re-clustering from scratch on
  every run — assign new images to existing clusters via nearest-centroid lookup,
  and periodically re-run full clustering to catch drift/merges.

## Known limitations

- The primary-face heuristic can mis-select when the true subject isn't the
  largest/most-central face in frame.
- `min_cluster_size=2` (this repo's demo-tuned default) provides essentially no
  noise-filtering power — must be raised for real deployments.
- A true singleton identity (only 1 photo of someone) can never form an HDBSCAN
  cluster (`min_cluster_size` is always ≥ 2), so it will always land in `unknown` —
  a real gap for datasets where some people have only one photo.
- Confidence scores are relative similarity measures, not calibrated
  probabilities.

## Project structure

```
project/
├── main.py              # CLI entrypoint + shared run_pipeline() orchestration
├── app.py                # Streamlit web UI (same run_pipeline())
├── detector.py           # FaceDetector + primary-face selection
├── embeddings.py         # ArcFace embedding extraction (L2-normalized)
├── clustering.py         # HDBSCAN + Agglomerative fallback
├── confidence.py         # Centroid-cosine-similarity confidence scoring
├── visualization.py      # Per-cluster montage PNGs
├── utils.py              # FaceRecord, I/O, logging
├── eval.py               # ARI/homogeneity/purity vs. filename ground truth
├── requirements.txt
├── DEMO_SCRIPT.md
├── input/                # your images go here (gitignored)
├── data/uploads/         # Streamlit-uploaded images persist here (gitignored)
└── output/               # generated: person_NNN/, unknown/, results.csv, visualizations/
```

## Future improvements

- Pose-quality-aware confidence discounting for extreme profile/occluded faces.
- FAISS-backed incremental clustering for streaming/growing datasets.
- GPU batch inference for large-scale throughput.
- Active-learning review UI for the `unknown` bucket (human-in-the-loop merge/split).
- Multi-frame subject tracking to disambiguate the primary-subject-vs-bystander
  problem more robustly than a single-frame bbox heuristic.
- Calibrated confidence (e.g. via a held-out verification set) instead of raw
  rescaled cosine similarity.
