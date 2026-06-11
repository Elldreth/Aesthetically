# Aesthetically

*(formerly imghotornot)*

A local, single-user image rating app for building a **personal aesthetic model**:
rate images fast, accumulate labels in SQLite, and (later phases) train a
taste scorer that pre-sorts everything for you.

## Run

```
run.bat
```

Creates `.venv` on first run, starts the server, opens http://127.0.0.1:8787.

## Rating

| Key | Action |
|---|---|
| `d` | Yay (like) |
| `w` | Maybe |
| `a` | Nay (dislike) |
| `space` | Skip (requeued for later) |
| `x` | Exclude — corrupt/off-topic, distinct from dislike |
| `z` | Undo last rating |

Images preload, sessions resume automatically (the queue is simply
"everything without a current label"), and the HUD tracks progress and
rate-per-minute. Once a taste model exists the queue defaults to
**most-uncertain-first** (active learning: those labels teach the model the
most), and your vote is followed by the model's score so you can watch it
learn. Other views:

- **grid ⊞** — bulk triage: thumbnails sorted by predicted score,
  multi-select + `a`/`w`/`d`, one-click **retrain** (seconds).
- **rank ⚔** — tournament mode: 6 of your liked images, click the best;
  every click records 5 pairwise comparisons for Bradley-Terry ranking
  (`/api/rankings`).

## Browser extension (rate the web)

Load `extension/` via `chrome://extensions` → Developer mode → *Load
unpacked* (the app must be running). Hover any image ≥120px on any page and
press `a`/`w`/`d` — the extension fetches the image *in page context* (your
cookies apply, so login-walled and hotlink-protected gallery images work)
and saves bytes + URLs + rating to `data/ingested/` via `POST /api/ingest`.
Re-rating the same image elsewhere dedupes by content hash.

## Layout

- `app/` — FastAPI server (`app.main:app`), schema, ingestion, static UI
- `app/artifex_client.py` — client for the [Artifex](../Artifex) SDXL sidecar
  (generation / LoRA training / dataset QA), used by later phases
- `data/aesthetically.db` — SQLite database (WAL). Labels are **append-only
  events**; images are identified by SHA-256 of file bytes; files stay where
  they are on disk.
- `migrate_folders.py` — one-time, non-destructive import of the v1
  folder-as-label layout (`matches/`, `maybes/`, `dislike/`, `liked_images/`,
  `found/`). Idempotent; never moves or deletes files.

## v1 legacy (retired, kept for reference)

`main.py` (Tkinter rater), `imgsearch.py`, `create_yolo.py`, `inference.py`,
`best.pt` and the label folders. The folders were imported into the DB on
2026-06-11 (6,754 unique images: 532 liked / 1,674 disliked / 13 maybe /
4,535 unlabeled). Don't delete them until you're confident in the DB.

## Studio — the Artifex closed loop

With [Artifex](../Artifex) running on `:7860` (override via `ARTIFEX_URL`),
the **studio ✨** view closes the loop:

- **Best-of-N**: type a prompt, generate N seeds, and your taste model
  ranks them — every generation is ingested with full metadata (prompt,
  checkpoint, seed) and scored, and your 👍/👎 on the results feeds the
  next retrain.
- **Taste LoRA**: one click curates your top-loved images (Bradley-Terry
  rank → taste score, then greedy max-min diversity selection in embedding
  space) and submits a style-LoRA job to Artifex. Runs are logged in
  `training_runs` with a dataset fingerprint.
- **Eval**: probe prompts (your own highest-scored prompts by default) ×
  fixed seeds, generated with and without the LoRA, scored on four axes —
  taste, prompt adherence (SigLIP text↔image), style fidelity (training-set
  centroid), and seed diversity (overfit detector). Results land in
  `eval_results` for cross-run comparison.

## ML pipeline

```
python -m app.embed      # SigLIP2-so400m embeddings for new images (GPU)
python -m app.dedupe     # phash near-dup detection (hamming 0 = identical)
python -m app.taste      # retrain the taste head + rescore (seconds)
python -m app.export --format imagefolder --out exports/run1
```

Training uses a cluster-aware train/val split (whole embedding-similarity
clusters held out) so same-prompt SD families never leak across the split.

## Roadmap

1. ~~FastAPI + SQLite + keyboard rater + folder migration~~ ← done
2. ~~SigLIP2 embeddings, taste model, grid triage, dedup, exports~~ ← done
3. ~~Active-learning queue, Chrome extension, tournament mode~~ ← done
4. Artifex closed loop: best-of-N filtering, taste-LoRA training, LoRA
   hyperparameter search with the taste model as objective
