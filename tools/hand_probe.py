"""Hand-quality probe — step 1: detect & crop hands.

Experiment (not part of the app): can we score 'good hands' from your taste?
This step runs Artifex's anime hand detector over a sample of your images,
crops each detected hand, and writes the crops + a manifest + a montage so we
can eyeball detection quality before investing in labeling/scoring.

    python -m tools.hand_probe --sample 400 --conf 0.3

Reads the Artifex YOLO model file read-only; does not touch the Artifex repo.
Outputs to data/hand_probe/.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
from PIL import Image

from app.db import DATA_DIR, get_conn

# A hand detector PER STYLE — an anime detector flops on photos and vice-versa.
#   anime     → Artifex's YOLO anime hand-seg model
#   realistic → MediaPipe HandLandmarker (purpose-built for real hands)
ANIME_MODEL = r"D:\repos\Artifex\models\ultralytics\anime_Hand_seg.pt"
OUT_DIR = DATA_DIR / "hand_probe"
CROPS_DIR = OUT_DIR / "crops"
MP_TASK = OUT_DIR / "hand_landmarker.task"


def _load_detectors(conf: float) -> dict:
    """Each style → (backend, model). Missing models are skipped, not mis-routed."""
    dets = {}
    if os.path.isfile(ANIME_MODEL):
        from ultralytics import YOLO
        dets["anime"] = ("yolo", YOLO(ANIME_MODEL))
        print(f"detector[anime] = yolo:{os.path.basename(ANIME_MODEL)}")
    if MP_TASK.is_file():
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MP_TASK)),
            num_hands=8, min_hand_detection_confidence=conf)
        dets["realistic"] = ("mediapipe", vision.HandLandmarker.create_from_options(opts))
        print("detector[realistic] = mediapipe:hand_landmarker")
    return dets


def _detect(backend, model, path, pil, conf) -> list[tuple]:
    """Return [(xyxy pixel box, confidence), ...] for one image."""
    W, H = pil.size
    if backend == "yolo":
        res = model.predict(path, conf=conf, verbose=False, device="cpu")[0]
        if res.boxes is None:
            return []
        return list(zip(res.boxes.xyxy.cpu().numpy(),
                        (float(c) for c in res.boxes.conf.cpu().numpy())))
    # mediapipe: 21 normalised landmarks per hand → tight bbox * (W,H)
    import mediapipe as mp
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=np.ascontiguousarray(np.array(pil)))
    result = model.detect(mp_img)
    out = []
    for i, lms in enumerate(result.hand_landmarks or []):
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        box = (min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H)
        cf = result.handedness[i][0].score if result.handedness else 1.0
        out.append((box, float(cf)))
    return out


def _sample_images(n: int, seed: int = 0) -> list[tuple[int, str, str]]:
    """Random tagged images that still exist on disk, with their style so each
    can go to the matching detector."""
    with get_conn() as db:
        rows = db.execute(
            "SELECT i.id, s.location, st.style FROM images i"
            " JOIN image_sources s ON s.image_id = i.id AND s.kind = 'local'"
            " JOIN image_styles st ON st.image_id = i.id"
            " GROUP BY i.id"
        ).fetchall()
    triples = [(r["id"], r["location"], r["style"]) for r in rows if os.path.isfile(r["location"])]
    random.Random(seed).shuffle(triples)
    return triples[:n]


def _expand(box, w, h, pad=0.6):
    """Pad a box by `pad` on each side and square it up, clamped to the image."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nx1, ny1 = max(0, cx - side / 2), max(0, cy - side / 2)
    nx2, ny2 = min(w, cx + side / 2), min(h, cy + side / 2)
    return int(nx1), int(ny1), int(nx2), int(ny2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min-px", type=int, default=40, help="ignore hand boxes smaller than this")
    ap.add_argument("--max-crops", type=int, default=600)
    ap.add_argument("--crop-size", type=int, default=256)
    args = ap.parse_args()

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    detectors = _load_detectors(args.conf)

    sample = _sample_images(args.sample)
    by_style = {}
    for _, _, st in sample:
        by_style[st] = by_style.get(st, 0) + 1
    print(f"scanning {len(sample)} tagged images "
          f"({', '.join(f'{k}:{v}' for k, v in by_style.items())}) conf>={args.conf}…")

    manifest, n_with_hands, n_crops = [], 0, 0
    for idx, (image_id, path, style) in enumerate(sample, 1):
        if n_crops >= args.max_crops:
            break
        det = detectors.get(style)
        if det is None:
            continue  # no detector for this style
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                W, H = im.size
                found = _detect(det[0], det[1], path, im, args.conf)
                if found:
                    n_with_hands += 1
                for j, (box, cf) in enumerate(found):
                    bw, bh = box[2] - box[0], box[3] - box[1]
                    if min(bw, bh) < args.min_px:
                        continue
                    crop = im.crop(_expand(box, W, H)).resize(
                        (args.crop_size, args.crop_size), Image.LANCZOS)
                    fname = f"{image_id}_{j}.png"
                    crop.save(CROPS_DIR / fname)
                    manifest.append({"crop": fname, "image_id": image_id, "style": style,
                                     "box": [int(v) for v in box], "conf": round(float(cf), 3)})
                    n_crops += 1
        except Exception as e:
            print(f"  skip {image_id}: {type(e).__name__}: {e}")
            continue
        if idx % 50 == 0:
            print(f"  {idx}/{len(sample)} scanned · {n_crops} hand crops so far")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # montage of up to 48 random crops for a visual quality check
    sel = random.Random(1).sample(manifest, min(48, len(manifest)))
    cols, cell = 8, 128
    rows = (len(sel) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (20, 20, 24))
    for i, m in enumerate(sel):
        thumb = Image.open(CROPS_DIR / m["crop"]).resize((cell, cell))
        sheet.paste(thumb, ((i % cols) * cell, (i // cols) * cell))
    sheet.save(OUT_DIR / "montage.png")

    rate = 100 * n_with_hands / max(1, min(len(sample), idx))
    print(f"\n{n_crops} hand crops from {n_with_hands} images "
          f"({rate:.0f}% of scanned had >=1 hand). manifest + montage in {OUT_DIR}")


if __name__ == "__main__":
    main()
