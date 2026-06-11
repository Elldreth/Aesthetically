"""SigLIP 2 image embeddings, cached in the embeddings table.

Embed once, train heads in seconds forever after. Run as a script to backfill
every image that doesn't have a vector yet:

    python -m app.embed [--batch 32]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import threading
import time

import numpy as np
from PIL import Image

from .db import get_conn

MODEL_NAME = os.environ.get("AESTH_EMBED_MODEL", "google/siglip2-so400m-patch16-384")

_model = None
_processor = None
_device = None
_load_lock = threading.Lock()


def _load_model():
    global _model, _processor, _device
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if _device == "cuda" else torch.float32
        model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(_device).eval()
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
        _model = model  # assign last: _model set means fully ready


def embed_pil(images: list[Image.Image]) -> np.ndarray:
    """L2-normalized float32 embeddings for a batch of PIL images."""
    import torch

    _load_model()
    inputs = _processor(images=images, return_tensors="pt").to(_device)
    with torch.no_grad():
        feats = _model.get_image_features(**inputs)
    if not torch.is_tensor(feats):  # newer transformers returns an output object
        feats = feats.pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()


def text_features(texts: list[str]) -> np.ndarray:
    """L2-normalized SigLIP text embeddings (same space as images — dual tower)."""
    import torch

    _load_model()
    inputs = _processor(text=texts, return_tensors="pt", padding="max_length",
                        truncation=True).to(_device)
    with torch.no_grad():
        feats = _model.get_text_features(**inputs)
    if not torch.is_tensor(feats):
        feats = feats.pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()


def load_vectors(db: sqlite3.Connection, model: str = MODEL_NAME,
                 image_ids: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(ids, matrix) of cached embeddings, optionally restricted to image_ids."""
    rows = db.execute(
        "SELECT image_id, dim, vec FROM embeddings WHERE model = ? ORDER BY image_id",
        (model,),
    ).fetchall()
    if image_ids is not None:
        want = set(image_ids)
        rows = [r for r in rows if r["image_id"] in want]
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    ids = np.array([r["image_id"] for r in rows], dtype=np.int64)
    mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32, count=r["dim"]) for r in rows])
    return ids, mat


def _local_path(db: sqlite3.Connection, image_id: int) -> str | None:
    for r in db.execute(
        "SELECT location FROM image_sources WHERE image_id = ? AND kind = 'local'",
        (image_id,),
    ):
        if os.path.isfile(r["location"]):
            return r["location"]
    return None


def backfill(batch_size: int = 32, progress: dict | None = None, cancel=None) -> None:
    db = get_conn()
    todo = [
        r["id"] for r in db.execute(
            "SELECT i.id FROM images i WHERE NOT EXISTS"
            " (SELECT 1 FROM embeddings e WHERE e.image_id = i.id AND e.model = ?)"
            " ORDER BY i.id",
            (MODEL_NAME,),
        )
    ]
    print(f"{len(todo)} images to embed with {MODEL_NAME}")
    if progress is not None:
        progress["total"] = len(todo)
        progress["done"] = 0
    if not todo:
        return
    _load_model()
    t0 = time.time()
    done = 0
    for start in range(0, len(todo), batch_size):
        if cancel is not None and cancel.is_set():
            return
        chunk = todo[start:start + batch_size]
        pils, ids = [], []
        for image_id in chunk:
            path = _local_path(db, image_id)
            if not path:
                continue
            try:
                with Image.open(path) as img:
                    img.draft("RGB", (512, 512))   # downscale big originals on decode
                    pils.append(img.convert("RGB"))
                ids.append(image_id)
            except Exception as e:
                print(f"  skip {image_id}: {type(e).__name__}")
        if not pils:
            continue
        vecs = embed_pil(pils)
        db.executemany(
            "INSERT OR REPLACE INTO embeddings (image_id, model, dim, vec) VALUES (?, ?, ?, ?)",
            [(i, MODEL_NAME, v.shape[0], v.tobytes()) for i, v in zip(ids, vecs)],
        )
        db.commit()
        done += len(ids)
        if progress is not None:
            progress["done"] = start + len(chunk)
        if done % (batch_size * 10) < batch_size:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(todo)}  ({rate:.1f} img/s, ~{(len(todo)-done)/rate:.0f}s left)")
    print(f"done: {done} embedded in {time.time() - t0:.0f}s")
    db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    backfill(batch_size=args.batch)
