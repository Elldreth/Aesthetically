"""Near-duplicate detection for queue hiding: perceptual hash.

    python -m app.dedupe [--phash-dist 4] [--cos <sim>]

Fills images.phash, then groups images whose phash Hamming distance crosses
the threshold — these are pixel-near-identical renders worth rating only once.
Each image in a group maps to the group's canonical (lowest-id) image in
near_dups. Nothing is deleted; the queue simply hides non-canonical members.

Embedding-cosine grouping (--cos) is OFF by default: SigLIP cosine ~0.99 still
conflates "identical render" with "same content, different aesthetics", and
the latter must stay individually ratable. Train/val leakage control does NOT
rely on this table — taste.py re-clusters embeddings at 0.92 on its own.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from PIL import Image

from .db import get_conn
from .embed import MODEL_NAME, load_vectors


def fill_phashes(db, progress: dict | None = None, cancel=None) -> None:
    import imagehash

    todo = db.execute(
        "SELECT i.id, s.location FROM images i"
        " JOIN image_sources s ON s.image_id = i.id AND s.kind = 'local'"
        " WHERE i.phash IS NULL GROUP BY i.id"
    ).fetchall()
    print(f"{len(todo)} images need phash")
    if progress is not None:
        progress["total"] = len(todo)
        progress["done"] = 0
    t0 = time.time()
    for n, row in enumerate(todo, 1):
        if cancel is not None and cancel.is_set():
            db.commit()
            return
        if not os.path.isfile(row["location"]):
            continue
        try:
            with Image.open(row["location"]) as img:
                h = str(imagehash.phash(img))
        except Exception as e:
            print(f"  skip {row['id']}: {type(e).__name__}")
            continue
        db.execute("UPDATE images SET phash = ? WHERE id = ?", (h, row["id"]))
        if n % 50 == 0:
            db.commit()  # commit often — embed.py writes concurrently
        if progress is not None:
            progress["done"] = n
        if n % 500 == 0:
            print(f"  {n}/{len(todo)} ({n / (time.time() - t0):.0f}/s)")
    db.commit()


class _UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def find_groups(db, phash_dist: int, cos_threshold: float | None) -> None:
    uf = _UnionFind()
    method: dict[int, str] = {}
    score: dict[int, float] = {}

    # phash pass — 64-bit hashes, brute-force pairs in numpy
    rows = db.execute("SELECT id, phash FROM images WHERE phash IS NOT NULL").fetchall()
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    hashes = np.array([int(r["phash"], 16) for r in rows], dtype=np.uint64)
    n_phash = 0
    xor = hashes[:, None] ^ hashes[None, :]
    # popcount via byte lookup table
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    dist = np.zeros(xor.shape, dtype=np.uint16)
    v = xor.copy()
    for _ in range(8):
        dist += lut[np.bitwise_and(v, np.uint64(0xFF)).astype(np.uint8)]
        v >>= np.uint64(8)
    ii, jj = np.where(np.triu(dist <= phash_dist, k=1))
    for a, b, d in zip(ids[ii], ids[jj], dist[ii, jj]):
        uf.union(int(a), int(b))
        n_phash += 1
        for x in (int(a), int(b)):
            method[x] = "phash"
            score[x] = float(d)
    print(f"phash pairs (hamming<={phash_dist}): {n_phash}")

    # optional embedding pass — cosine over cached vectors
    eids, mat = (np.empty(0), np.empty((0, 0))) if cos_threshold is None \
        else load_vectors(db, MODEL_NAME)
    n_cos = 0
    if len(eids):
        sims = mat @ mat.T
        ii, jj = np.where(np.triu(sims >= cos_threshold, k=1))
        for a, b, s in zip(eids[ii], eids[jj], sims[ii, jj]):
            uf.union(int(a), int(b))
            n_cos += 1
            for x in (int(a), int(b)):
                method[x] = "phash+embedding" if method.get(x) == "phash" else "embedding"
                score[x] = float(s)
        print(f"embedding pairs (cos>={cos_threshold}): {n_cos}")

    db.execute("DELETE FROM near_dups")
    groups: dict[int, list[int]] = {}
    for x in uf.parent:
        groups.setdefault(uf.find(x), []).append(x)
    n_members = 0
    for canonical, members in groups.items():
        for m in members:
            if m == canonical:
                continue
            db.execute(
                "INSERT OR REPLACE INTO near_dups (image_id, canonical_id, method, score)"
                " VALUES (?, ?, ?, ?)",
                (m, canonical, method.get(m, "phash"), score.get(m)),
            )
            n_members += 1
    db.commit()
    print(f"groups: {len(groups)}, non-canonical members hidden from queue: {n_members}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # default 0: on SD seed-cluster data even hamming 2 conflates aesthetic
    # variants (clean vs artifacted renders) that must be rated separately
    ap.add_argument("--phash-dist", type=int, default=0)
    ap.add_argument("--cos", type=float, default=None,
                    help="also group by embedding cosine (off by default; see docstring)")
    args = ap.parse_args()
    db = get_conn()
    fill_phashes(db)
    find_groups(db, args.phash_dist, args.cos)
    db.close()
