"""Score arbitrary images against the latest trained taste head.

Lazy: nothing heavy loads until the first score request. Thread-guarded so
FastAPI's threadpool can't double-load SigLIP.
"""
from __future__ import annotations

import json
import threading

import numpy as np
from PIL import Image

from .db import get_conn
from .embed import MODEL_NAME, embed_pil
from .taste import MODELS_DIR

_lock = threading.Lock()


def latest_head() -> dict | None:
    versions = sorted(MODELS_DIR.glob("taste_v*.json"),
                      key=lambda p: int(p.stem.split("_v")[1]))
    if not versions:
        return None
    return json.loads(versions[-1].read_text(encoding="utf-8"))


def score_images(pils: list[Image.Image], head: dict | None = None,
                 return_vecs: bool = False):
    """P(like) for PIL images; optionally also the embeddings (for caching)."""
    head = head or latest_head()
    if head is None:
        raise RuntimeError("no trained taste head — run app.taste first")
    with _lock:
        vecs = embed_pil([p.convert("RGB") for p in pils])
    logits = vecs @ np.array(head["coef"], dtype=np.float32) + head["intercept"]
    scores = (1.0 / (1.0 + np.exp(-logits))).tolist()
    return (scores, vecs) if return_vecs else scores


def store_embedding_and_score(image_id: int, vec: np.ndarray, score: float,
                              head_name: str) -> None:
    with get_conn() as db:
        db.execute(
            "INSERT OR REPLACE INTO embeddings (image_id, model, dim, vec) VALUES (?, ?, ?, ?)",
            (image_id, MODEL_NAME, vec.shape[0], vec.tobytes()),
        )
        db.execute(
            "INSERT OR REPLACE INTO predictions (image_id, model, score) VALUES (?, ?, ?)",
            (image_id, head_name, float(score)),
        )
