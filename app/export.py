"""Dataset exports for training runs.

    python -m app.export --format imagefolder --out exports/run1
    python -m app.export --format csv --out exports/labels.csv

imagefolder: HF-datasets layout — images copied into the output folder with a
metadata.jsonl (file_name, label, soft_score, prompt, model_hash, split).
Loads directly via `datasets.load_dataset("imagefolder", data_dir=...)`.

csv: one row per labeled image (path, label, score, prompt, model_hash, split).

Splits are CLUSTER-AWARE (same grouping as taste.py) so near-dup SD families
never straddle train/val. Every export is recorded in exports/export_items
with the parameters that produced it, so any snapshot is reproducible.
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import json
import shutil
from pathlib import Path

import numpy as np

from .db import get_conn
from .embed import MODEL_NAME, load_vectors
from .taste import CLUSTER_COS, SEED, VAL_FRACTION, _clusters


def _rows(db):
    return db.execute(
        """SELECT i.id, i.prompt, i.model_hash, c.value,
                  (SELECT location FROM image_sources s
                   WHERE s.image_id = i.id AND s.kind = 'local' LIMIT 1) AS path,
                  (SELECT score FROM predictions p
                   WHERE p.image_id = i.id AND p.model LIKE 'taste:%' LIMIT 1) AS score
           FROM images i
           JOIN current_labels c ON c.image_id = i.id AND c.kind = 'binary'
           WHERE NOT EXISTS (SELECT 1 FROM current_labels e
                             WHERE e.image_id = i.id AND e.kind = 'exclude')
             AND NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)
           ORDER BY i.id"""
    ).fetchall()


def _splits(db, image_ids: list[int]) -> dict[int, str]:
    """train/val per image, whole similarity clusters assigned together."""
    ids, mat = load_vectors(db, MODEL_NAME, image_ids=image_ids)
    if not len(ids):
        # no embeddings yet — fall back to per-image split, deterministic
        rng = np.random.default_rng(SEED)
        return {i: ("val" if rng.random() < VAL_FRACTION else "train") for i in image_ids}
    roots = _clusters(ids, mat, db)
    rng = np.random.default_rng(SEED)
    unique_roots = np.unique(roots)
    rng.shuffle(unique_roots)
    val_roots = set(unique_roots[: max(1, int(len(unique_roots) * VAL_FRACTION))].tolist())
    split = {int(i): ("val" if r in val_roots else "train") for i, r in zip(ids, roots)}
    return {i: split.get(i, "train") for i in image_ids}


def export(fmt: str, out: Path) -> dict:
    db = get_conn()
    rows = [r for r in _rows(db) if r["path"]]
    split = _splits(db, [r["id"] for r in rows])
    label_names = {0.0: "dislike", 0.5: "maybe", 1.0: "like"}

    if fmt == "imagefolder":
        out.mkdir(parents=True, exist_ok=True)
        meta_path = out / "metadata.jsonl"
        with meta_path.open("w", encoding="utf-8") as meta:
            for r in rows:
                src = Path(r["path"])
                dst_name = f"{r['id']}{src.suffix.lower()}"
                shutil.copy2(src, out / dst_name)
                meta.write(json.dumps({
                    "file_name": dst_name,
                    "label": label_names[r["value"]],
                    "soft_score": r["value"],
                    "taste_score": r["score"],
                    "prompt": r["prompt"],
                    "model_hash": r["model_hash"],
                    "split": split[r["id"]],
                }) + "\n")
    elif fmt == "csv":
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv_mod.writer(f)
            w.writerow(["image_id", "path", "label", "soft_score", "taste_score",
                        "prompt", "model_hash", "split"])
            for r in rows:
                w.writerow([r["id"], r["path"], label_names[r["value"]], r["value"],
                            r["score"], r["prompt"], r["model_hash"], split[r["id"]]])
    else:
        raise SystemExit(f"unknown format: {fmt}")

    cur = db.execute(
        "INSERT INTO exports (name, format, query_json) VALUES (?, ?, ?)",
        (out.name, fmt, json.dumps({
            "cluster_cos": CLUSTER_COS, "val_fraction": VAL_FRACTION, "seed": SEED,
            "embed_model": MODEL_NAME, "n": len(rows),
        })),
    )
    db.executemany(
        "INSERT INTO export_items (export_id, image_id, label_value) VALUES (?, ?, ?)",
        [(cur.lastrowid, r["id"], r["value"]) for r in rows],
    )
    db.commit()
    counts = {"train": 0, "val": 0}
    for r in rows:
        counts[split[r["id"]]] += 1
    db.close()
    return {"export_id": cur.lastrowid, "format": fmt, "out": str(out),
            "items": len(rows), **counts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["imagefolder", "csv"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(json.dumps(export(args.format, Path(args.out)), indent=2))
