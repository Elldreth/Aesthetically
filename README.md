# Aesthetically

*(formerly imghotornot)*

Rate images fast, teach a model **your** taste, and let it pre-sort everything
you haven't seen yet. Aesthetically is a local, single-user web app for
building personal aesthetic models: label with three keys, retrain in seconds,
and (optionally) close the loop with SDXL generation and LoRA training.

## Quick start

Requirements: **Windows + Python 3.12** (3.11–3.13 should work), and an NVIDIA
GPU for the ML features (rating works on any machine).

```
git clone https://github.com/Elldreth/Aesthetically.git
cd Aesthetically
run.bat
```

`run.bat` creates `.venv`, installs dependencies, starts the server, and opens
`http://127.0.0.1:8787`. For GPU embedding, install the CUDA torch wheel
first:

```
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### Getting images in

- **Local folders**: adapt `migrate_folders.py` (it registers files in place —
  nothing is moved or copied), or POST to `/api/ingest`.
- **The web**: install the browser extension (below).
- **Generated**: the Studio view, with Artifex running (below).

## Rating

| Key | Action |
|---|---|
| `d` | Yay (like) |
| `w` | Maybe |
| `a` | Nay (dislike) |
| `space` | Skip — requeued for later |
| `x` | Remove — broken/off-topic, *not* the same as dislike |
| `z` | Undo last rating |
| `?` | Keyboard help |

Sessions resume automatically — the queue is simply "everything without a
current label." Once a taste model exists, the queue defaults to
**most-uncertain-first**: those labels teach the model the most, and your vote
is followed by the model's score so you can watch it learn.

**Views:** *Rate* (single image, keyboard-first) · *Grid* (bulk triage sorted
by model score, one-click retrain) · *Tournament* (best-of-6 among your likes;
each click records 5 pairwise comparisons for Bradley-Terry ranking) ·
*Studio* (generation + LoRA training via Artifex).

## The ML pipeline

```
.venv\Scripts\python -m app.embed     # SigLIP2 embeddings for new images (GPU, ~25 img/s)
.venv\Scripts\python -m app.dedupe    # perceptual-hash duplicate detection
.venv\Scripts\python -m app.taste     # (re)train the taste head + rescore everything
.venv\Scripts\python -m app.export --format imagefolder --out exports\run1
```

Run `embed` after adding images (the server never embeds by itself), `dedupe`
once per big import, `taste` whenever you've added labels (also available as
the **Retrain** button in Grid — it takes seconds because embeddings are
cached). Exports: HF-datasets `imagefolder` + `metadata.jsonl`, or CSV — both
with **cluster-aware train/val splits** (whole embedding-similarity clusters
held out, so near-duplicate SD families never leak across the split and
inflate your validation numbers).

Dedup note: only hamming-distance-0 (structurally identical) images are
hidden from the queue. On SD seed-cluster data, even distance-2 pairs are
*aesthetic variants* — one clean, one artifacted — which is exactly the signal
a taste model needs labeled.

## Browser extension

`chrome://extensions` → Developer mode → **Load unpacked** → select
`extension/`. Then open the extension's **options** and paste the API token
(from `data/token.txt`). On any page: hover an image ≥120px, press
`a`/`w`/`d`. Pixels are captured from the page when possible (no extra
request); otherwise fetched — with credentials only for same-site images.
Content-hash dedup means the same image rated on two sites is one record with
two sources.

## Studio — the Artifex closed loop

With [Artifex](https://github.com/Elldreth) (a self-contained SDXL FastAPI
sidecar) running on `:7860` (`ARTIFEX_URL` to override):

- **Best-of-N**: generate N seeds, your taste model ranks them; every
  generation is ingested with full metadata (prompt, checkpoint, seed) and
  your 👍/👎 feeds the next retrain.
- **Taste LoRA**: one click curates your top-loved images (Bradley-Terry rank
  → taste score → greedy max-min diversity selection in embedding space) and
  submits a style-LoRA job. Runs are logged in `training_runs` with a dataset
  fingerprint.
- **Eval**: probe prompts × fixed seeds, with/without the LoRA, scored on four
  axes — taste, prompt adherence (SigLIP text↔image), style fidelity
  (training-set centroid), seed diversity. Everything lands in `eval_results`.

Everything degrades gracefully when Artifex is offline — the Studio shows a
callout and the rest of the app is unaffected.

## Architecture

```
app/main.py      FastAPI server: rating API, image/thumb serving, studio proxy
app/schema.sql   SQLite schema — the core invariant: labels are APPEND-ONLY
                 events; "current" state is the latest label per (image, kind)
                 via the current_labels view. Undo = delete one row.
app/embed.py     SigLIP2 image+text embeddings, cached as BLOBs per model
app/taste.py     logistic head on frozen embeddings; cluster-aware val split
app/scorer.py    score arbitrary images against the latest head
app/dedupe.py    perceptual-hash near-dup groups (queue hiding)
app/studio.py    the Artifex loop: best-of-N, dataset curation, LoRA, eval
app/export.py    reproducible dataset snapshots (exports/export_items)
extension/       Chrome MV3: rate any image on any page into the local API
data/            SQLite DB, thumbnails, ingested/generated images, taste
                 models, API token — BACK THIS FOLDER UP; images stay where
                 they are on disk and are never modified.
```

Images are identified by SHA-256 of file bytes; the same image known from
disk and a URL is one record with multiple `image_sources` rows.

API reference: interactive docs at `http://127.0.0.1:8787/docs`. Label values:
`1` like, `0.5` maybe, `0` dislike; `exclude` is a separate label kind.
Mutating endpoints require the `X-Aesth-Token` header (or the cookie the UI
sets) — token in `data/token.txt`.

### Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `ARTIFEX_URL` | `http://127.0.0.1:7860` | Artifex sidecar |
| `AESTH_EMBED_MODEL` | `google/siglip2-so400m-patch16-384` | embedding model |
| `AESTH_ALLOWED_HOSTS` | `127.0.0.1:8787,localhost:8787` | Host-header allowlist |

### Security posture

Designed to bind to **loopback only**. Mutating requests need a per-install
token (SameSite=Strict cookie / header), the Host allowlist blocks DNS
rebinding, image decoding is capped at 64MP, and uploads are size-limited. Do
not expose uvicorn directly to a network; if you must share, put auth + TLS in
a reverse proxy in front — and know that the app currently has no
multi-account concept.

## Tests

```
.venv\Scripts\python -m pytest tests -q
```

No GPU, no Artifex, no real data needed — tests run against a temp DB with
fake embeddings and a fake Artifex client.

## v1 legacy

`main.py` (Tkinter rater), `imgsearch.py`, `create_yolo.py`, `inference.py`,
`best.pt`, and the label folders are the original 2023 tool, kept for
reference. Their labels were imported on 2026-06-11 (6,754 unique images).
