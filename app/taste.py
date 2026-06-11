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


def train() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    db = get_conn()
    labeled = {
        r["image_id"]: r["value"]
        for r in db.execute(
            "SELECT image_id, value FROM current_labels WHERE kind = 'binary'"
        )
    }
    ids, mat = load_vectors(db, MODEL_NAME, image_ids=list(labeled))
    if len(ids) < 50:
        raise SystemExit(f"only {len(ids)} labeled+embedded images — run app.embed first")
    y_soft = np.array([labeled[int(i)] for i in ids], dtype=np.float32)

    # cluster-aware split (maybes ride along but train as soft 0.5 -> excluded
    # from the hard logistic fit; they still get predictions)
    roots = _clusters(ids, mat, db)
    rng = np.random.default_rng(SEED)
    unique_roots = np.unique(roots)
    rng.shuffle(unique_roots)
    n_val = max(1, int(len(unique_roots) * VAL_FRACTION))
    val_roots = set(unique_roots[:n_val].tolist())
    is_val = np.array([r in val_roots for r in roots])
    hard = y_soft != 0.5

    X_tr, y_tr = mat[~is_val & hard], (y_soft[~is_val & hard] == 1.0).astype(int)
    X_va, y_va = mat[is_val & hard], (y_soft[is_val & hard] == 1.0).astype(int)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    fit_s = time.time() - t0

    p_va = clf.predict_proba(X_va)[:, 1]
    metrics = {
        "train_n": int(len(y_tr)),
        "val_n": int(len(y_va)),
        "val_clusters": int(n_val),
        "total_clusters": int(len(unique_roots)),
        "val_accuracy": round(float(accuracy_score(y_va, p_va >= 0.5)), 4),
        "val_auc": round(float(roc_auc_score(y_va, p_va)), 4) if len(set(y_va)) > 1 else None,
        "fit_seconds": round(fit_s, 2),
        "embed_model": MODEL_NAME,
    }

    # refit on ALL hard labels before deployment (val measured generalization above)
    clf.fit(mat[hard], (y_soft[hard] == 1.0).astype(int))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = 1 + max(
        (int(p.stem.split("_v")[1]) for p in MODELS_DIR.glob("taste_v*.json")), default=0
    )
    name = f"taste:v{version}"
    out = {
        "name": name,
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "metrics": metrics,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (MODELS_DIR / f"taste_v{version}.json").write_text(json.dumps(out), encoding="utf-8")

    # score EVERY embedded image
    all_ids, all_mat = load_vectors(db, MODEL_NAME)
    logits = all_mat @ np.array(out["coef"], dtype=np.float32) + out["intercept"]
    scores = 1.0 / (1.0 + np.exp(-logits))
    db.execute("DELETE FROM predictions WHERE model LIKE 'taste:%'")
    db.executemany(
        "INSERT OR REPLACE INTO predictions (image_id, model, score) VALUES (?, ?, ?)",
        [(int(i), name, float(s)) for i, s in zip(all_ids, scores)],
    )
    db.commit()
    db.close()
    metrics["model"] = name
    metrics["scored"] = int(len(all_ids))
    return metrics


if __name__ == "__main__":
    m = train()
    print(json.dumps(m, indent=2))
