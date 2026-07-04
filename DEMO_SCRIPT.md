# Demo Video Script (2–3 minutes)

**On-screen surface:** the Streamlit app (`streamlit run app.py`), recorded live.

---

### 1. Problem statement (~20s)

"Given an unorganized folder of photos — no labels, no idea how many people are in
it — I need to automatically group every photo of the same person together, and
give each photo a confidence score for that grouping. The dataset I'm demoing with
makes this genuinely hard: these are full-body snapshots from a bowling alley, with
dim, colored lighting, some motion blur, and — this is the tricky part — every
photo has 2 to 4 *other* people visible in the background. So face detection alone
isn't enough; the system has to figure out who the actual subject is in each shot."

*(Show the raw input images briefly — point out the bystanders.)*

### 2. Pipeline walkthrough (~30s)

"The pipeline has four stages. First, face detection — I'm using RetinaFace via
InsightFace, which finds every face in the frame. Second, primary-face selection —
for each photo I pick the largest detected face, since the subject fills more of
the frame than someone in the background. Third, alignment and embedding — each
selected face gets warped to a canonical pose using 5-point landmarks, then passed
through ArcFace to get a 512-dimensional embedding that represents that person's
identity. Fourth, clustering — HDBSCAN groups all the embeddings into identities
automatically."

*(In the Streamlit UI: upload the sample images, point at the bbox overlay in a
result thumbnail showing the selected primary face.)*

### 3. Model selection rationale (~30s)

"I chose InsightFace's bundled `buffalo_l` pack over wiring together separate
detector and recognition models, because it gives detection, alignment, and ArcFace
embedding in one coherent, CPU-friendly ONNX Runtime pipeline — no GPU required,
which matters for a demo running on a laptop. ArcFace specifically uses an additive
angular margin loss, which produces embeddings with tight same-identity clustering
and wide separation between different identities — exactly the property clustering
needs downstream."

### 4. Clustering choice (~30s)

"For clustering, I used HDBSCAN instead of k-means, because k-means needs you to
specify the number of clusters up front — and the whole point here is we don't know
how many people are in the dataset. HDBSCAN figures that out automatically, and
anything that doesn't fit a cluster gets marked as noise, which maps directly onto
an 'unknown' bucket. One important tuning detail: this demo set only has 2 photos
per person, so I had to lower HDBSCAN's minimum cluster size to 2 — the default of
5 would have marked everything as noise. That's a parameter you'd raise back up on
a denser real-world dataset."

*(Show the sidebar parameter controls in the Streamlit app.)*

### 5. Confidence scoring (~20s)

"Each image's confidence score is the cosine similarity between its embedding and
the average embedding — the centroid — of its assigned cluster, rescaled to a 0-to-1
range. Here you can see both photos of this person score around 0.96 confidence,
even with the purple lighting and the person behind them in frame."

*(Show the CSV / confidence badges live in the UI.)*

### 6. Final output (~20s)

"The end result: three folders, one per person, each correctly containing that
person's two photos, plus a CSV with filename, cluster ID, and confidence for every
image. Running this against known ground truth gives a perfect Adjusted Rand Index
and cluster purity of 1.0 — all three identities recovered correctly despite the
lighting, blur, and background bystanders."

*(Show the output folder tree and the eval metrics tab.)*

### 7. Future improvements (~20s)

"Looking ahead: for datasets in the hundreds of thousands of images, I'd swap the
current precomputed distance matrix for an approximate nearest-neighbor index like
FAISS, move inference to GPU, and add incremental clustering so new photos get
assigned to existing identities without a full re-cluster. I'd also want a
human-in-the-loop review step for the 'unknown' bucket, and better handling of
extreme profile angles and heavy occlusion."

---

**Total runtime target: ~2m50s.**
