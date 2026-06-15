"""Style tagging: split the collection along the anime↔realistic axis.

A mixed corpus (anime + realistic) muddies one global taste model and makes
LoRAs fight the base checkpoint. Tagging each image's style lets rating,
training, and scoring run per-style. Auto-tags come from the SigLIP text axis
(the one axis that separates cleanly); hand corrections are preserved.
"""
from __future__ import annotations

import numpy as np

from .db import conn, get_conn
from .embed import MODEL_NAME, load_vectors, text_features

# higher cos to [0] than [1] ⇒ anime-leaning
_AXIS = [
    "an anime illustration, 2d drawing, cel-shaded artwork",
    "a realistic photograph of a real person or scene",
]
STYLES = ("anime", "realistic")


def classify_styles(progress: dict | None = None, cancel=None) -> dict:
    """Auto-tag every embedded image as anime/realistic. Never overwrites a
    'manual' tag. Returns counts."""
    if progress is not None:
        progress.update(phase="loading", done=0, total=0)
    with conn() as db:
        ids, mat = load_vectors(db, MODEL_NAME)
    if not len(ids):
        return {"anime": 0, "realistic": 0, "tagged": 0}

    axis = text_features(_AXIS)
    score = mat @ axis[0] - mat @ axis[1]          # >0 anime, <=0 realistic
    rows = [(int(i), "anime" if s > 0 else "realistic", round(float(abs(s)), 4))
            for i, s in zip(ids, score)]

    if progress is not None:
        progress.update(phase="tagging", total=len(rows), done=0)
    db = get_conn()
    try:
        for n in range(0, len(rows), 1000):
            if cancel is not None and cancel.is_set():
                break
            chunk = rows[n:n + 1000]
            # ON CONFLICT … WHERE source='auto' preserves manual corrections
            db.executemany(
                "INSERT INTO image_styles (image_id, style, source, margin)"
                " VALUES (?, ?, 'auto', ?)"
                " ON CONFLICT(image_id) DO UPDATE SET style=excluded.style,"
                " margin=excluded.margin WHERE image_styles.source='auto'",
                chunk,
            )
            db.commit()
            if progress is not None:
                progress["done"] = min(n + 1000, len(rows))
    finally:
        db.close()
    return counts()


def set_style(image_ids: list[int], style: str) -> int:
    """Manually set/override the style for images (wins over auto)."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    with conn() as db:
        db.executemany(
            "INSERT INTO image_styles (image_id, style, source, margin)"
            " VALUES (?, ?, 'manual', NULL)"
            " ON CONFLICT(image_id) DO UPDATE SET style=excluded.style, source='manual'",
            [(i, style) for i in image_ids],
        )
    return len(image_ids)


def counts() -> dict:
    with conn() as db:
        out = {s: 0 for s in STYLES}
        for r in db.execute("SELECT style, count(*) AS n FROM image_styles GROUP BY style"):
            out[r["style"]] = r["n"]
        out["untagged"] = db.execute(
            "SELECT count(*) AS n FROM images i"
            " WHERE NOT EXISTS (SELECT 1 FROM image_styles s WHERE s.image_id = i.id)"
        ).fetchone()["n"]
        out["manual"] = db.execute(
            "SELECT count(*) AS n FROM image_styles WHERE source='manual'").fetchone()["n"]
    return out
