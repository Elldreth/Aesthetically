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


def hand_clause(hand_filter: str | None, alias: str = "i") -> str:
    """SQL fragment for a hand-quality filter on image `alias`.
    'good' = only images tagged good hands; 'not_bad' = exclude tagged bad."""
    if hand_filter == "good":
        return (f" AND EXISTS (SELECT 1 FROM hand_labels hl"
                f" WHERE hl.image_id = {alias}.id AND hl.label = 1)")
    if hand_filter == "not_bad":
        return (f" AND NOT EXISTS (SELECT 1 FROM hand_labels hl"
                f" WHERE hl.image_id = {alias}.id AND hl.label = 0)")
    return ""


def _candidates(db, unlabeled_only: bool, style: str | None = None,
                hand_filter: str | None = None) -> list[dict]:
    label_filter = """AND NOT EXISTS (SELECT 1 FROM current_labels c
                       WHERE c.image_id = i.id AND c.kind IN ('binary','exclude'))""" \
        if unlabeled_only else ""
    style_filter = (" AND EXISTS (SELECT 1 FROM image_styles st"
                    " WHERE st.image_id = i.id AND st.style = ?)") if style else ""
    params = (style,) if style else ()
    rows = db.execute(f"""
        SELECT i.id, p.score,
               (SELECT location FROM image_sources s WHERE s.image_id = i.id
                AND s.kind = 'local' LIMIT 1) AS path
        FROM images i
        JOIN predictions p ON p.image_id = i.id AND p.model LIKE 'taste:%'
        WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)
        {label_filter}{style_filter}{hand_clause(hand_filter)}
        ORDER BY p.score DESC""", params).fetchall()
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


def run_select(out: Path, *, top: int | None = None, min_score: float | None = None,
               buckets: bool = False, unlabeled_only: bool = False,
               style: str | None = None, hand_filter: str | None = None,
               mode: str = "copy", progress: dict | None = None) -> dict:
    """Shared by the CLI and the web UI. Returns a summary dict."""
    progress = progress if progress is not None else {}
    db = get_conn()
    cands = _candidates(db, unlabeled_only, style=style, hand_filter=hand_filter)
    db.close()
    if top:
        cands = cands[:top]
    elif min_score is not None:
        cands = [c for c in cands if c["score"] >= min_score]
    progress.update(total=len(cands), done=0, phase="exporting")
    n = 0
    for c in cands:
        name = f"{c['score']:.3f}_{c['id']}{Path(c['path']).suffix.lower()}"
        if buckets:
            bucket = f"{min(int(c['score'] * 10), 9) / 10:.1f}+"
            dst = out / bucket / name
        else:
            dst = out / name
        _transfer(c["path"], dst, mode)
        n += 1
        progress["done"] = n
    progress["phase"] = "done"
    return {"count": n, "out": str(out),
            "min_score_taken": round(cands[-1]["score"], 3) if cands else None}


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

    res = run_select(Path(args.out), top=args.top, min_score=args.min_score,
                     buckets=args.buckets, unlabeled_only=args.unlabeled_only,
                     mode=args.mode)
    print(f"{args.mode}d {res['count']} images -> {res['out']}"
          + (f" (score >= {res['min_score_taken']})" if res["min_score_taken"] else ""))


if __name__ == "__main__":
    main()
