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
rate-per-minute.

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

## Roadmap

1. ~~FastAPI + SQLite + keyboard rater + folder migration~~ ← done
2. SigLIP2 embeddings, grid triage, near-dup detection, dataset exports
   (HF imagefolder + metadata.jsonl)
3. Active-learning queue, Chrome extension for web galleries, pairwise
   "best of screen" mode
4. Artifex closed loop: best-of-N filtering, taste-LoRA training, LoRA
   hyperparameter search with the taste model as objective
