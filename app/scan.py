"""Ephemeral folder scoring: predict taste for ANY folder without touching
the collection. Nothing is written to the database — results live in memory
(and optionally a CSV / an exported top-N folder).

    python -m app.scan D:\\some\\folder [--csv out.csv] [--top 50 --out D:\\keepers]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from .ingest import DECODE_ERRORS, iter_image_files
from .scorer import latest_head, score_images

BATCH = 16


def run_scan(folder: Path, progress: dict, *, recursive: bool = True) -> list[dict]:
    """Score every image in a folder. Mutates progress for polling; returns
    [{path, score}] sorted best-first (also stored as progress['results'])."""
    head = latest_head()
    if head is None:
        raise RuntimeError("no trained taste model yet")
    files = iter_image_files(folder, recursive=recursive)
    progress.update(total=len(files), done=0, state="scoring", model=head["name"])
    results: list[dict] = []
    for start in range(0, len(files), BATCH):
        chunk = files[start:start + BATCH]
        pils, kept = [], []
        for f in chunk:
            try:
                with Image.open(f) as img:
                    pils.append(img.convert("RGB"))
                kept.append(f)
            except DECODE_ERRORS + (OSError,):
                pass
        if pils:
            scores = score_images(pils, head)
            results.extend(
                {"path": str(f), "score": round(float(s), 4)}
                for f, s in zip(kept, scores)
            )
        progress["done"] = min(start + BATCH, len(files))
    results.sort(key=lambda r: -r["score"])
    progress.update(state="done", results=results)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--csv", help="write path,score CSV here")
    ap.add_argument("--top", type=int, help="copy the N best to --out")
    ap.add_argument("--out", help="destination for --top")
    args = ap.parse_args()

    results = run_scan(Path(args.folder), {})
    for r in results[:20]:
        print(f"{r['score']:.3f}  {r['path']}")
    if len(results) > 20:
        print(f"... and {len(results) - 20} more")
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
