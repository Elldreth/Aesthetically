"""Act on taste-model predictions: pull the best images out of the collection.

    python -m app.select --top 200 --out D:\\keepers          # copy top 200
    python -m app.select --min-score 0.8 --out D:\\keepers    # copy all >= 0.8
    python -m app.select --buckets --out D:\\sorted           # group into score deciles
    python -m app.select --top 50 --unlabeled-only --link     # hardlink, skip rated

Copies by default (originals never touched); --move and --link available.
Source of scores: the latest taste:% predictions (run Add Folder / app.embed
+ app.taste first so everything is scored).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .db import get_conn


def _candidates(db, unlabeled_only: bool) -> list[dict]:
    label_filter = """AND NOT EXISTS (SELECT 1 FROM current_labels c
                       WHERE c.image_id = i.id AND c.kind IN ('binary','exclude'))""" \
        if unlabeled_only else ""
    rows = db.execute(f"""
        SELECT i.id, p.score,
               (SELECT location FROM image_sources s WHERE s.image_id = i.id
                AND s.kind = 'local' LIMIT 1) AS path
        FROM images i
        JOIN predictions p ON p.image_id = i.id AND p.model LIKE 'taste:%'
        WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)
        {label_filter}
        ORDER BY p.score DESC""").fetchall()
    return [dict(r) for r in rows if r["path"] and os.path.isfile(r["path"])]


def _transfer(src: str, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "move":
        shutil.move(src, dst)
    elif mode == "link":
        try:
            os.link(src, dst)          # hardlink: instant, same-volume only
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="destination folder")
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--top", type=int, help="take the N highest-scored images")
    pick.add_argument("--min-score", type=float, help="take everything at/above this score")
    ap.add_argument("--buckets", action="store_true",
                    help="group into score-decile subfolders (0.9+, 0.8+, ...)")
    ap.add_argument("--unlabeled-only", action="store_true",
                    help="only images you haven't rated (pure predictions)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--move", action="store_const", const="move", dest="mode")
    mode.add_argument("--link", action="store_const", const="link", dest="mode",
                      help="hardlink instead of copy (no extra disk, same volume)")
    ap.set_defaults(mode="copy")
    args = ap.parse_args()
    if not (args.top or args.min_score or args.buckets):
        ap.error("pick --top N, --min-score X, or --buckets")

    db = get_conn()
    cands = _candidates(db, args.unlabeled_only)
    db.close()
    if args.top:
        cands = cands[: args.top]
    elif args.min_score is not None:
        cands = [c for c in cands if c["score"] >= args.min_score]

    out = Path(args.out)
    n = 0
    for c in cands:
        name = f"{c['score']:.3f}_{c['id']}{Path(c['path']).suffix.lower()}"
        if args.buckets:
            bucket = f"{min(int(c['score'] * 10), 9) / 10:.1f}+"
            dst = out / bucket / name
        else:
            dst = out / name
        _transfer(c["path"], dst, args.mode)
        n += 1
    print(f"{args.mode}d {n} images -> {out}"
          + (f" (score >= {cands[-1]['score']:.3f})" if cands else ""))


if __name__ == "__main__":
    main()
