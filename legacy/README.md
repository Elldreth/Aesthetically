# Legacy (v1)

The original 2023 "imghotornot" tool, kept for reference. **Nothing in `app/`
imports any of this** — it's history, not a dependency.

- `v1_rater.py` — the original Tkinter rater (was `main.py` at the repo root;
  renamed to avoid confusion with `app/main.py`, the current server).
- `imgsearch.py` — crawled folders for large Stable Diffusion PNGs and sorted
  them by model hash.
- `create_yolo.py` / `inference.py` / `best.pt` — the abandoned
  YOLO-as-classifier experiment (whole-image bounding boxes). Replaced by the
  SigLIP2 embedding + taste-head pipeline.
- `migrate_folders.py` — one-time importer that brought the v1 folder-as-label
  layout into the SQLite database on 2026-06-11. Already run; here for the record.
