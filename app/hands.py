"""Anime hand-quality scoring.

A logistic head over SigLIP hand-crop embeddings, built on the user's rule
"liked => good hands": positives are hands cropped from LIKED anime images (free,
no labeling), negatives are hands from images explicitly marked bad (labels
kind='hand', value=0). Every anime image is scored by the WORST of its detected
hands, so one bad hand sinks the image. Realistic images are good by
construction (real photos) and are not scored.

Hand detection uses Artifex's anime YOLO seg model (path via ANIME_HAND_MODEL).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from PIL import Image

from .db import DATA_DIR, conn, get_conn
from .embed import MODEL_NAME, embed_pil

MODELS_DIR = DATA_DIR / "models"
ANIME_HAND_MODEL = os.environ.get(
    "ANIME_HAND_MODEL", r"D:\repos\Artifex\models\ultralytics\anime_Hand_seg.pt")
MIN_PX, PAD, DET_CONF = 40, 0.6, 0.3
MIN_PER_CLASS = 20

_yolo = None


def _detector():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(ANIME_HAND_MODEL)
    return _yolo


def _expand(box, w, h, pad=PAD):
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
            int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))


def crop_hands(path: str) -> list[Image.Image]:
    """Detected hand crops for one image (empty if none / unreadable)."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size
            res = _detector().predict(np.array(im), conf=DET_CONF, verbose=False, device="cpu")[0]
            if res.boxes is None:
                return []
            out = []
            for box in res.boxes.xyxy.cpu().numpy():
                if min(box[2] - box[0], box[3] - box[1]) < MIN_PX:
                    continue
                out.append(im.crop(_expand(box, W, H)).resize((256, 256), Image.LANCZOS))
            return out
    except Exception:
        return []


def _embed_image_hands(rows) -> tuple[np.ndarray, np.ndarray]:
    """Crop+embed hands for (id, path) rows. Returns (vectors, source image ids)."""
    vecs, ids = [], []
    for r in rows:
        crops = crop_hands(r["location"])
        if not crops:
            continue
        vecs.append(embed_pil(crops))
        ids.extend([r["id"]] * len(crops))
    if not vecs:
        return np.empty((0, 1152), np.float32), np.empty(0, np.int64)
    return np.vstack(vecs), np.array(ids, np.int64)


def _latest_version() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return max((int(p.stem.split("_v")[1]) for p in MODELS_DIR.glob("hand_v*.json")), default=0)


def latest_head() -> dict | None:
    v = _latest_version()
    if not v:
        return None
    return json.loads((MODELS_DIR / f"hand_v{v}.json").read_text())


def _good_query():
    return ("SELECT i.id, s.location FROM images i"
            " JOIN current_labels c ON c.image_id=i.id AND c.kind='binary' AND c.value=1.0"
            " JOIN image_styles st ON st.image_id=i.id AND st.style='anime'"
            " JOIN image_sources s ON s.image_id=i.id AND s.kind='local'"
            " WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id=i.id)"
            " GROUP BY i.id")


def _bad_query():
    return ("SELECT i.id, s.location FROM images i"
            " JOIN hand_labels hl ON hl.image_id=i.id AND hl.label=0"
            " JOIN image_sources s ON s.image_id=i.id AND s.kind='local'"
            " GROUP BY i.id")


def train(max_good: int = 400, progress: dict | None = None, cancel=None) -> dict:
    """Train the hand head: good = liked-anime hands, bad = marked-bad hands."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    with conn() as db:
        good_rows = db.execute(_good_query()).fetchall()
        bad_rows = db.execute(_bad_query()).fetchall()
    import random
    random.Random(0).shuffle(good_rows := list(good_rows))
    if progress is not None:
        progress.update(phase="cropping good", total=len(bad_rows) + min(len(good_rows), max_good * 2))

    Xb, gb = _embed_image_hands(bad_rows)
    # harvest good (liked-anime) hand crops until roughly balanced with bad
    Xg_parts, gg_parts, n = [], [], 0
    target = min(max(len(gb), MIN_PER_CLASS * 2), max_good)
    for r in good_rows:
        if cancel is not None and cancel.is_set():
            break
        if n >= target:
            break
        crops = crop_hands(r["location"])
        if not crops:
            continue
        Xg_parts.append(embed_pil(crops))
        gg_parts.extend([r["id"]] * len(crops))
        n += len(crops)
    Xg = np.vstack(Xg_parts) if Xg_parts else np.empty((0, 1152), np.float32)
    gg = np.array(gg_parts, np.int64)

    n_good, n_bad = len(Xg), len(Xb)
    if n_good < MIN_PER_CLASS or n_bad < MIN_PER_CLASS:
        raise RuntimeError(
            f"need >={MIN_PER_CLASS} of each: good(liked-anime hands)={n_good}, "
            f"bad(marked-bad hands)={n_bad}. Mark more bad hands in the grid.")

    X = np.vstack([Xg, Xb])
    y = np.concatenate([np.ones(n_good), np.zeros(n_bad)])
    groups = np.concatenate([gg, gb])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    n_splits = min(5, len(set(groups)), n_good, n_bad)
    proba = cross_val_predict(clf, X, y, cv=GroupKFold(n_splits=n_splits),
                              groups=groups, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    clf.fit(X, y)

    version = _latest_version() + 1
    name = f"hand:v{version}"
    head = {"name": name, "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]), "model": MODEL_NAME,
            "metrics": {"val_auc": round(auc, 3), "n_good": n_good, "n_bad": n_bad},
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    (MODELS_DIR / f"hand_v{version}.json").write_text(json.dumps(head), encoding="utf-8")
    return {"model": name, "val_auc": round(auc, 3), "n_good": n_good, "n_bad": n_bad}


def _prob(vecs: np.ndarray, head: dict) -> np.ndarray:
    z = vecs @ np.array(head["coef"]) + head["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def score_all(progress: dict | None = None, cancel=None, limit: int | None = None) -> dict:
    """Score every anime image by its worst detected hand; store in hand_scores."""
    head = latest_head()
    if not head:
        raise RuntimeError("no hand model yet — train first")
    with conn() as db:
        rows = db.execute(
            "SELECT i.id, s.location FROM images i"
            " JOIN image_styles st ON st.image_id=i.id AND st.style='anime'"
            " JOIN image_sources s ON s.image_id=i.id AND s.kind='local'"
            " WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id=i.id)"
            " GROUP BY i.id" + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    if progress is not None:
        progress.update(phase="scoring", total=len(rows), done=0)
    scored = 0
    for n, r in enumerate(rows, 1):
        if cancel is not None and cancel.is_set():
            break
        crops = crop_hands(r["location"])
        if crops:
            probs = _prob(embed_pil(crops), head)
            worst = float(np.min(probs))
            with conn() as db:
                db.execute(
                    "INSERT INTO hand_scores (image_id, score, n_hands, model, updated_at)"
                    " VALUES (?, ?, ?, ?, datetime('now'))"
                    " ON CONFLICT(image_id) DO UPDATE SET score=excluded.score,"
                    " n_hands=excluded.n_hands, model=excluded.model, updated_at=excluded.updated_at",
                    (r["id"], worst, len(crops), head["name"]))
            scored += 1
        if progress is not None:
            progress["done"] = n
    return {"model": head["name"], "scored": scored, "anime_total": len(rows)}
