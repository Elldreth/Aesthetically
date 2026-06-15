"""Hand probe v2 — use 'liked => good hands' to build the good class for free.

The user never likes an image with bad hands, so hand crops from LIKED anime
images are good examples (no manual labeling). Bad examples are the bad-anime
crops already marked in the probe. Both classes are anime + hand-crops only, so
this measures hand quality without the style confound and without new bboxing.

    python -m tools.hand_likes_probe
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from app.db import DATA_DIR, get_conn
from app.embed import MODEL_NAME, embed_pil
from tools.hand_probe import ANIME_MODEL, _expand

DIR = DATA_DIR / "hand_probe"
GOOD_DIR = DIR / "good_likes"


def harvest_good(limit: int, conf: float = 0.3) -> list[tuple]:
    """Detect + crop hands from liked anime images = good-hand examples."""
    from ultralytics import YOLO

    GOOD_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(ANIME_MODEL)
    with get_conn() as db:
        rows = db.execute(
            "SELECT i.id, s.location FROM images i"
            " JOIN current_labels c ON c.image_id = i.id AND c.kind='binary' AND c.value=1.0"
            " JOIN image_styles st ON st.image_id = i.id AND st.style='anime'"
            " JOIN image_sources s ON s.image_id = i.id AND s.kind='local'"
            " WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)"
            " GROUP BY i.id"
        ).fetchall()
    pairs = [(r["id"], r["location"]) for r in rows if os.path.isfile(r["location"])]
    random.Random(0).shuffle(pairs)
    crops, scanned = [], 0
    for image_id, path in pairs:
        if len(crops) >= limit:
            break
        scanned += 1
        try:
            res = model.predict(path, conf=conf, verbose=False, device="cpu")[0]
        except Exception:
            continue
        if res.boxes is None:
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size
            for j, box in enumerate(res.boxes.xyxy.cpu().numpy()):
                if min(box[2] - box[0], box[3] - box[1]) < 40:
                    continue
                f = GOOD_DIR / f"{image_id}_{j}.png"
                im.crop(_expand(box, W, H)).resize((256, 256), Image.LANCZOS).save(f)
                crops.append((f, image_id))
                if len(crops) >= limit:
                    break
    print(f"harvested {len(crops)} good-hand crops from {scanned} liked anime images")
    return crops


def load_bad_anime() -> list[tuple]:
    manifest = {m["crop"]: m for m in json.loads((DIR / "manifest.json").read_text())}
    img_labels = json.loads((DIR / "labels.json").read_text())
    return [(DIR / "crops" / crop, m["image_id"]) for crop, m in manifest.items()
            if m["style"] == "anime" and img_labels.get(str(m["image_id"])) == "bad"]


def main():
    bad = load_bad_anime()
    good = harvest_good(limit=max(len(bad), 150))
    if len(good) < 30 or len(bad) < 30:
        print(f"need more data (good={len(good)}, bad={len(bad)})")
        return
    print(f"good (liked anime): {len(good)} crops · bad (labeled anime): {len(bad)} crops")

    files = [f for f, _ in good] + [f for f, _ in bad]
    y = np.array([1] * len(good) + [0] * len(bad))
    groups = np.array([g for _, g in good] + [g for _, g in bad])

    X = []
    for i in range(0, len(files), 32):
        X.append(embed_pil([Image.open(f).convert("RGB") for f in files[i:i + 32]]))
    X = np.vstack(X)
    print(f"embedded {len(files)} crops, {X.shape[1]}-d ({MODEL_NAME.split('/')[-1]})")

    n_splits = min(5, len(set(groups)), int(y.sum()), int((1 - y).sum()))
    proba = cross_val_predict(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        X, y, cv=GroupKFold(n_splits=n_splits), groups=groups, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, (proba >= 0.5).astype(int))
    base = max(int(y.sum()), int((1 - y).sum())) / len(y)
    print(f"\n  ANIME good-vs-bad (likes-as-good, hand crops only)")
    print(f"  n={len(y)} ({n_splits}-fold grouped CV)")
    print(f"  AUC      {auc:.3f}   (0.5 = no signal, 1.0 = perfect)")
    print(f"  accuracy {acc:.3f}   (majority baseline {base:.3f})")
    verdict = ("STRONG - the likes-as-good approach works; scale it" if auc >= 0.80 else
               "MODERATE - usable as a soft filter; keypoints would add" if auc >= 0.70 else
               "WEAK - SigLIP crops can't separate good/bad anime hands")
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
