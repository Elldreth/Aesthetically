"""The personal taste model: logistic head on frozen SigLIP 2 embeddings.

    python -m app.taste            # train, report metrics, write predictions

Training is seconds on CPU. Validation uses a CLUSTER-AWARE split: images are
grouped by embedding similarity (plus near-dup groups) and whole clusters go
to train or val — never split a cluster, or SD same-prompt families leak and
val accuracy inflates 5-15 points.

The trained head is saved to data/models/taste_vN.json (weights + metrics);
P(like) for every embedded image goes to the predictions table as 'taste:vN'.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .db import DATA_DIR, get_conn
from .embed import MODEL_NAME, load_vectors

MODELS_DIR = DATA_DIR / "models"
CLUSTER_COS = 0.92  # expert-recommended grouping threshold for leakage control
VAL_FRACTION = 0.2
SEED = 1337
MIN_LABELS = 40     # per (style) model; below this there isn't enough to train
STYLES = ("anime", "realistic")


def _clusters(ids: np.ndarray, mat: np.ndarray, db) -> np.ndarray:
    """Cluster id per image: connected components of cosine>=CLUSTER_COS, seeded
    with near_dups groups. Returns an array of cluster roots aligned with ids."""
    index = {int(v): k for k, v in enumerate(ids)}
    parent = np.arange(len(ids))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    sims = mat @ mat.T
    ii, jj = np.where(np.triu(sims >= CLUSTER_COS, k=1))
    for a, b in zip(ii, jj):
        union(int(a), int(b))
    for r in db.execute("SELECT image_id, canonical_id FROM near_dups"):
        a, b = index.get(r["image_id"]), index.get(r["canonical_id"])
        if a is not None and b is not None:
            union(a, b)
    return np.array([find(i) for i in range(len(ids))])


def train(style: str | None = None) -> dict:
    """Train a taste head. With style ('anime'/'realistic') it trains only on
    that style's labels, names the model 'taste:<style>:vN', and scores only
    that style's images. With style=None it trains one global model on
    everything ('taste:vN'). Raises RuntimeError if too few labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    if style is not None and style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")

    db = get_conn()
    if style:
        labeled = {r["image_id"]: r["value"] for r in db.execute(
            "SELECT c.image_id, c.value FROM current_labels c"
            " JOIN image_styles s ON s.image_id = c.image_id AND s.style = ?"
            " WHERE c.kind = 'binary'", (style,))}
        tag, file_prefix, del_pattern = f"taste:{style}", f"taste_{style}", f"taste:{style}:%"
    else:
        labeled = {r["image_id"]: r["value"] for r in db.execute(
            "SELECT image_id, value FROM current_labels WHERE kind = 'binary'")}
        tag, file_prefix, del_pattern = "taste", "taste", "taste:v%"

    ids, mat = load_vectors(db, MODEL_NAME, image_ids=list(labeled))
    y_soft = np.array([labeled[int(i)] for i in ids], dtype=np.float32)
    hard = y_soft != 0.5
    y_hard = (y_soft[hard] == 1.0).astype(int)
    label = style or "all"
    if int(hard.sum()) < MIN_LABELS or len(set(y_hard.tolist())) < 2:
        db.close()
        raise RuntimeError(
            f"{label}: only {int(hard.sum())} usable labels (need {MIN_LABELS}+ "
            "with both like and dislike) — rate more")

    # cluster-aware split (maybes train as soft 0.5 -> excluded from the hard fit)
    roots = _clusters(ids, mat, db)
    rng = np.random.default_rng(SEED)
    unique_roots = np.unique(roots)
    rng.shuffle(unique_roots)
    n_val = max(1, int(len(unique_roots) * VAL_FRACTION))
    val_roots = set(unique_roots[:n_val].tolist())
    is_val = np.array([r in val_roots for r in roots])

    X_tr, y_tr = mat[~is_val & hard], (y_soft[~is_val & hard] == 1.0).astype(int)
    X_va, y_va = mat[is_val & hard], (y_soft[is_val & hard] == 1.0).astype(int)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    p_va = clf.predict_proba(X_va)[:, 1]
    metrics = {
        "style": style, "train_n": int(len(y_tr)), "val_n": int(len(y_va)),
        "val_clusters": int(n_val), "total_clusters": int(len(unique_roots)),
        "val_accuracy": round(float(accuracy_score(y_va, p_va >= 0.5)), 4),
        "val_auc": round(float(roc_auc_score(y_va, p_va)), 4) if len(set(y_va)) > 1 else None,
        "fit_seconds": round(fit_s, 2), "embed_model": MODEL_NAME,
    }

    clf.fit(mat[hard], y_hard)   # refit on all hard labels for deployment

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = 1 + max((int(p.stem.split("_v")[1])
                       for p in MODELS_DIR.glob(f"{file_prefix}_v*.json")), default=0)
    name = f"{tag}:v{version}"
    out = {"name": name, "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
           "metrics": metrics, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (MODELS_DIR / f"{file_prefix}_v{version}.json").write_text(json.dumps(out), encoding="utf-8")

    # score the relevant images (this style's, or all for a global model)
    if style:
        score_ids = [r["image_id"] for r in db.execute(
            "SELECT image_id FROM image_styles WHERE style = ?", (style,))]
        score_ids, score_mat = load_vectors(db, MODEL_NAME, image_ids=score_ids)
    else:
        score_ids, score_mat = load_vectors(db, MODEL_NAME)
    logits = score_mat @ np.array(out["coef"], dtype=np.float32) + out["intercept"]
    scores = 1.0 / (1.0 + np.exp(-logits))
    db.execute(f"DELETE FROM predictions WHERE model LIKE '{del_pattern}'")
    db.executemany(
        "INSERT OR REPLACE INTO predictions (image_id, model, score) VALUES (?, ?, ?)",
        [(int(i), name, float(s)) for i, s in zip(score_ids, scores)],
    )
    db.commit()
    db.close()
    metrics["model"] = name
    metrics["scored"] = int(len(score_ids))
    return metrics


def train_styles() -> dict:
    """Retrain every style that has enough labels; score each style's images
    by its own model. Clears any legacy global predictions so the per-style
    models are authoritative."""
    db = get_conn()
    with db:
        db.execute("DELETE FROM predictions WHERE model LIKE 'taste:v%'")  # drop legacy global
    db.close()
    out = {}
    for style in STYLES:
        try:
            out[style] = train(style)
        except RuntimeError as e:
            out[style] = {"skipped": str(e)}
    return out


if __name__ == "__main__":
    print(json.dumps(train_styles(), indent=2))
