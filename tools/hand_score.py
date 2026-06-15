"""Hand-quality probe — step 3: can your good/bad hand labels be learned?

Embeds the hand crops you labeled (SigLIP, same model as the taste head),
then measures how well a logistic head predicts good-vs-bad on held-out crops.
Splits are grouped by SOURCE IMAGE so multiple crops from one picture can't
leak across train/val.

    python -m tools.hand_score

Reads data/hand_probe/{manifest,labels}.json + crops; prints AUC/accuracy and
saves the trained head to data/hand_probe/hand_head.json.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

from app.db import DATA_DIR
from app.embed import MODEL_NAME, embed_pil

DIR = DATA_DIR / "hand_probe"


def main():
    manifest = {m["crop"]: m for m in json.loads((DIR / "manifest.json").read_text())}
    # labels.json is keyed by SOURCE IMAGE id ('good'/'bad'); propagate each
    # image's label to all of its hand crops.
    img_labels = json.loads((DIR / "labels.json").read_text())
    items = [(c, img_labels[str(m["image_id"])]) for c, m in manifest.items()
             if str(m["image_id"]) in img_labels and img_labels[str(m["image_id"])] in ("good", "bad")]
    if len(items) < 30:
        n_img = sum(1 for v in img_labels.values() if v in ("good", "bad"))
        print(f"only {len(items)} labeled crops (from {n_img} images) — label ~150+ images.")
        return

    crops = [c for c, _ in items]
    y = np.array([1 if v == "good" else 0 for _, v in items])
    groups = np.array([manifest[c]["image_id"] for c in crops])
    styles = np.array([manifest[c]["style"] for c in crops])
    n_good, n_bad = int(y.sum()), int((1 - y).sum())
    print(f"{len(items)} labeled crops · {n_good} good / {n_bad} bad "
          f"· {len(set(groups))} distinct source images")
    if n_good < 10 or n_bad < 10:
        print("need at least ~10 of each class for a stable estimate.")
        return

    # embed crops (batched) with the taste model's SigLIP
    X = []
    for i in range(0, len(crops), 32):
        pils = [Image.open(DIR / "crops" / c).convert("RGB") for c in crops[i:i + 32]]
        X.append(embed_pil(pils))
    X = np.vstack(X)
    print(f"embedded {X.shape[0]} crops → {X.shape[1]}-d ({MODEL_NAME.split('/')[-1]})")

    def evaluate(Xi, yi, gi, label):
        ng, nb = int(yi.sum()), int((1 - yi).sum())
        if ng < 8 or nb < 8 or len(set(gi)) < 3:
            print(f"\n  [{label}] too few labels ({ng} good / {nb} bad) — skipped")
            return None
        n_splits = min(5, len(set(gi)), ng, nb)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        proba = cross_val_predict(clf, Xi, yi, cv=GroupKFold(n_splits=n_splits),
                                  groups=gi, method="predict_proba")[:, 1]
        auc = roc_auc_score(yi, proba)
        acc = accuracy_score(yi, (proba >= 0.5).astype(int))
        base = max(ng, nb) / len(yi)
        verdict = ("STRONG — worth a real hand sub-score" if auc >= 0.80 else
                   "PROMISING — more labels / try keypoint signals" if auc >= 0.68 else
                   "WEAK — SigLIP crops miss it; lean on keypoint/finger-count")
        print(f"\n  [{label}] {len(yi)} crops · {ng} good / {nb} bad · {n_splits}-fold grouped CV")
        print(f"    AUC      {auc:.3f}   (0.5 = no signal, 1.0 = perfect)")
        print(f"    accuracy {acc:.3f}   (majority baseline {base:.3f})")
        print(f"    → {verdict}")
        return float(auc)

    overall_auc = evaluate(X, y, groups, "all styles")
    for st in ("anime", "realistic"):
        mask = styles == st
        if mask.sum():
            evaluate(X[mask], y[mask], groups[mask], st)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X, y)
    auc = overall_auc or 0.0
    (DIR / "hand_head.json").write_text(json.dumps(
        {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
         "model": MODEL_NAME, "n": len(y), "auc": round(float(auc), 3)}), encoding="utf-8")
    print(f"  saved head → {DIR / 'hand_head.json'}")


if __name__ == "__main__":
    main()
