"""Ephemeral folder scoring: predict taste for ANY folder without adding it to
the training collection.

Embeddings are cached by file content hash (scan_cache), so re-scoring a folder
is instant and a new taste model rescores from cache without re-running SigLIP.
Each scan is persisted (scans + scan_items) so you can reopen or export it later
— it survives server restarts and you can keep many scans around.

    python -m app.scan D:\\some\\folder [--csv out.csv] [--top 50 --out D:\\keepers]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .db import get_conn
from .embed import MODEL_NAME, embed_pil
from .ingest import DECODE_ERRORS, iter_image_files
from .scorer import latest_head

BATCH = 16


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _decode_downscaled(data: bytes) -> Image.Image:
    """Decode at reduced scale — SigLIP only needs 384px, so never hold a
    full-res 6000px image in memory just to shrink it."""
    img = Image.open(BytesIO(data))
    img.draft("RGB", (512, 512))           # fast partial JPEG decode (no-op for PNG)
    img = img.convert("RGB")
    img.thumbnail((512, 512))
    return img


def run_scan(folder: Path, scan_id: int, progress: dict, *,
             recursive: bool = True) -> list[dict]:
    """Score every image in a folder, caching embeddings by content hash and
    persisting results under scan_id. Mutates progress for status polling."""
    head = latest_head()
    if head is None:
        raise RuntimeError("no trained taste model yet")
    coef = np.array(head["coef"], dtype=np.float32)
    intercept = float(head["intercept"])
    files = iter_image_files(folder, recursive=recursive)
    progress.update(total=len(files), done=0, state="scoring",
                    model=head["name"], scan_id=scan_id)

    db = get_conn()
    results: list[dict] = []
    pending: list[tuple[str, str, Image.Image]] = []  # (path, sha, pil) cache-misses

    def flush() -> None:
        if not pending:
            return
        vecs = embed_pil([p for _, _, p in pending])
        rows = []
        for (path, sha, _), vec in zip(pending, vecs):
            rows.append((sha, MODEL_NAME, vec.shape[0], vec.tobytes()))
            score = float(_sigmoid(vec @ coef + intercept))
            results.append({"path": path, "sha256": sha, "score": score})
        db.executemany(
            "INSERT OR REPLACE INTO scan_cache (sha256, model, dim, vec) VALUES (?, ?, ?, ?)",
            rows,
        )
        db.commit()
        pending.clear()

    try:
        for n, f in enumerate(files, 1):
            try:
                data = f.read_bytes()
            except OSError:
                progress["done"] = n
                continue
            sha = hashlib.sha256(data).hexdigest()
            cached = db.execute(
                "SELECT vec, dim FROM scan_cache WHERE sha256 = ? AND model = ?",
                (sha, MODEL_NAME),
            ).fetchone()
            if cached:
                vec = np.frombuffer(cached["vec"], dtype=np.float32, count=cached["dim"])
                score = float(_sigmoid(vec @ coef + intercept))
                results.append({"path": str(f), "sha256": sha, "score": score})
            else:
                try:
                    pending.append((str(f), sha, _decode_downscaled(data)))
                except DECODE_ERRORS:
                    progress["done"] = n
                    continue
                if len(pending) >= BATCH:
                    flush()
            progress["done"] = n
        flush()

        results.sort(key=lambda r: -r["score"])
        db.executemany(
            "INSERT OR REPLACE INTO scan_items (scan_id, path, sha256, score)"
            " VALUES (?, ?, ?, ?)",
            [(scan_id, r["path"], r["sha256"], r["score"]) for r in results],
        )
        db.execute("UPDATE scans SET count = ? WHERE id = ?", (len(results), scan_id))
        db.commit()
    finally:
        db.close()

    progress.update(state="done", count=len(results))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--csv", help="write path,score CSV here")
    ap.add_argument("--top", type=int, help="copy the N best to --out")
    ap.add_argument("--out", help="destination for --top")
    args = ap.parse_args()

    folder = Path(args.folder)
    db = get_conn()
    cur = db.execute(
        "INSERT INTO scans (path, taste_model, embed_model) VALUES (?, ?, ?)",
        (str(folder), (latest_head() or {}).get("name"), MODEL_NAME),
    )
    scan_id = cur.lastrowid
    db.commit()
    db.close()

    results = run_scan(folder, scan_id, {})
    for r in results[:20]:
        print(f"{r['score']:.3f}  {r['path']}")
    if len(results) > 20:
        print(f"... and {len(results) - 20} more  (scan #{scan_id})")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "score"])
            w.writerows([(r["path"], r["score"]) for r in results])
        print(f"csv -> {args.csv}")
    if args.top and args.out:
        import shutil

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for r in results[: args.top]:
            src = Path(r["path"])
            shutil.copy2(src, out / f"{r['score']:.3f}_{src.name}")
        print(f"copied top {min(args.top, len(results))} -> {out}")


if __name__ == "__main__":
    main()
