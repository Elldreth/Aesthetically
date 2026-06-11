"""One-time, non-destructive migration of the v1 folder-as-label layout into the DB.

Reads (never moves/deletes):
  matches/       -> binary 1.0
  maybes/        -> binary 0.5
  dislike/       -> binary 0.0
  liked_images/  -> binary 1.0 (session-tagged 'migration:liked_images')
  found/         -> unlabeled; subfolder name recorded as model_hash, sibling
                    .txt (from imgsearch.py) recorded as prompt when missing
  input/         -> unlabeled

Conflict rule (same sha256 in multiple label folders): labels are inserted in
ascending strictness order (dislike < maybe < match) so the latest/strongest
wins in current_labels; every conflict is logged.

Idempotent: re-running adds nothing new (sha256-keyed images, unique source
rows, and migration labels are only inserted when that exact label is absent).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from app.db import get_conn
from app.ingest import iter_image_files, upsert_image

ROOT = Path(__file__).resolve().parent

# (folder, value, note) — insertion order implements the conflict rule.
LABEL_FOLDERS = [
    ("dislike", 0.0, None),
    ("maybes", 0.5, None),
    ("liked_images", 1.0, "liked_images"),
    ("matches", 1.0, None),
]
UNLABELED_FOLDERS = ["found", "input"]


def has_migration_label(db, image_id: int, value: float) -> bool:
    return db.execute(
        "SELECT 1 FROM labels WHERE image_id = ? AND kind = 'binary'"
        " AND value = ? AND source = 'migration'",
        (image_id, value),
    ).fetchone() is not None


def main() -> None:
    db = get_conn()
    t0 = time.time()
    seen_labels: dict[int, list[float]] = {}  # image_id -> values assigned this run
    stats = {"files": 0, "unreadable": 0, "labels": 0, "conflicts": 0}

    for folder, value, note in LABEL_FOLDERS:
        path = ROOT / folder
        if not path.is_dir():
            continue
        files = iter_image_files(path)
        print(f"[{folder}] {len(files)} image files -> label {value}")
        for f in files:
            stats["files"] += 1
            image_id = upsert_image(db, f)
            if image_id is None:
                stats["unreadable"] += 1
                print(f"  unreadable: {f}")
                continue
            prior = seen_labels.setdefault(image_id, [])
            if prior and value not in prior:
                stats["conflicts"] += 1
                print(f"  CONFLICT image {image_id} ({f.name}): {prior} then {value} — latest wins")
            if not has_migration_label(db, image_id, value):
                db.execute(
                    "INSERT INTO labels (image_id, kind, value, source) VALUES (?, 'binary', ?, 'migration')",
                    (image_id, value),
                )
                stats["labels"] += 1
            prior.append(value)
        db.commit()

    for folder in UNLABELED_FOLDERS:
        path = ROOT / folder
        if not path.is_dir():
            continue
        files = iter_image_files(path)
        print(f"[{folder}] {len(files)} image files -> unlabeled pool")
        for f in files:
            stats["files"] += 1
            image_id = upsert_image(db, f)
            if image_id is None:
                stats["unreadable"] += 1
                continue
            # imgsearch.py layout: found/<model_hash>/<name>.png + <name>.txt caption
            if folder == "found" and f.parent != path:
                db.execute(
                    "UPDATE images SET model_hash = COALESCE(model_hash, ?) WHERE id = ?",
                    (f.parent.name, image_id),
                )
                txt = f.with_suffix(".txt")
                if txt.is_file():
                    try:
                        prompt = txt.read_text(encoding="utf-8", errors="replace").strip()
                        if prompt:
                            db.execute(
                                "UPDATE images SET prompt = COALESCE(prompt, ?) WHERE id = ?",
                                (prompt, image_id),
                            )
                    except OSError:
                        pass
        db.commit()

    n_images = db.execute("SELECT count(*) FROM images").fetchone()[0]
    n_current = db.execute(
        "SELECT value, count(*) FROM current_labels WHERE kind='binary' GROUP BY value"
    ).fetchall()
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"  files scanned : {stats['files']}  (unreadable: {stats['unreadable']})")
    print(f"  unique images : {n_images}")
    print(f"  labels added  : {stats['labels']}  (conflicts: {stats['conflicts']})")
    for value, n in n_current:
        name = {0.0: "disliked", 0.5: "maybe", 1.0: "liked"}.get(value, value)
        print(f"  current {name}: {n}")
    db.close()


if __name__ == "__main__":
    sys.exit(main())
