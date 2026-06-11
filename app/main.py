"""Aesthetically — local FastAPI server: rating API + static UI + image serving."""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import DATA_DIR, get_conn
from .ingest import inspect_image

app = FastAPI(title="Aesthetically")

# the browser extension calls /api/ingest from chrome-extension:// origins
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome|moz)-extension://.*$",
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

VALID_BINARY = {0.0, 0.5, 1.0}


class LabelIn(BaseModel):
    image_id: int
    value: float  # 1 = yay, 0.5 = maybe, 0 = nay
    session_id: int | None = None


class ExcludeIn(BaseModel):
    image_id: int
    session_id: int | None = None


class UndoIn(BaseModel):
    session_id: int | None = None


class SessionIn(BaseModel):
    name: str | None = None


def conn():
    return get_conn()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/img/{image_id}")
def serve_image(image_id: int):
    with conn() as db:
        rows = db.execute(
            "SELECT location FROM image_sources WHERE image_id = ? AND kind = 'local'",
            (image_id,),
        ).fetchall()
    for row in rows:
        if os.path.isfile(row["location"]):
            return FileResponse(row["location"])
    raise HTTPException(404, "no readable local source for image")


_UNRATED = """
    FROM images i
    LEFT JOIN predictions p ON p.image_id = i.id AND p.model LIKE 'taste:%'
    WHERE NOT EXISTS (SELECT 1 FROM current_labels c
                      WHERE c.image_id = i.id AND c.kind IN ('binary','exclude'))
      AND NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)
"""
_QUEUE_ORDER = {
    "default": "i.id",
    "uncertain": "CASE WHEN p.score IS NULL THEN 1 ELSE 0 END, ABS(p.score - 0.5)",
    "best": "p.score IS NULL, p.score DESC",
    "worst": "p.score IS NULL, p.score ASC",
}


@app.get("/api/queue")
def queue(limit: int = 20, mode: str = "default"):
    """Unlabeled, non-excluded, non-duplicate images.

    mode=uncertain surfaces the active-learning sweet spot (score near 0.5);
    best/worst sort by the taste model's P(like)."""
    order = _QUEUE_ORDER.get(mode)
    if order is None:
        raise HTTPException(422, f"mode must be one of {sorted(_QUEUE_ORDER)}")
    with conn() as db:
        rows = db.execute(
            f"""SELECT i.id, i.width, i.height, i.prompt, i.model_hash, p.score
                {_UNRATED} ORDER BY {order} LIMIT ?""",
            (limit,),
        ).fetchall()
    return {"items": [dict(r) for r in rows]}


@app.get("/thumb/{image_id}")
def thumbnail(image_id: int):
    """Cached 320px WebP thumbnails for the grid view."""
    thumb_dir = DATA_DIR / "thumbs"
    thumb_path = thumb_dir / f"{image_id}.webp"
    if not thumb_path.is_file():
        with conn() as db:
            rows = db.execute(
                "SELECT location FROM image_sources WHERE image_id = ? AND kind = 'local'",
                (image_id,),
            ).fetchall()
        src = next((r["location"] for r in rows if os.path.isfile(r["location"])), None)
        if not src:
            raise HTTPException(404, "no readable local source for image")
        from PIL import Image

        thumb_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.thumbnail((320, 320))
            img.save(thumb_path, "WEBP", quality=80)
    return FileResponse(thumb_path, media_type="image/webp")


@app.get("/api/grid")
def grid(mode: str = "worst", limit: int = 60, offset: int = 0):
    """Paged unrated images for grid triage, sorted by taste score."""
    order = _QUEUE_ORDER.get(mode)
    if order is None:
        raise HTTPException(422, f"mode must be one of {sorted(_QUEUE_ORDER)}")
    with conn() as db:
        total = db.execute(f"SELECT count(*) AS n {_UNRATED}").fetchone()["n"]
        rows = db.execute(
            f"""SELECT i.id, p.score {_UNRATED} ORDER BY {order} LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
    return {"total": total, "items": [dict(r) for r in rows]}


class BulkLabelIn(BaseModel):
    image_ids: list[int]
    value: float
    session_id: int | None = None


@app.post("/api/labels/bulk")
def bulk_label(body: BulkLabelIn):
    if body.value not in VALID_BINARY:
        raise HTTPException(422, "value must be 1, 0.5 or 0")
    if not body.image_ids:
        return {"labeled": 0}
    with conn() as db:
        db.executemany(
            "INSERT INTO labels (image_id, kind, value, source, session_id)"
            " VALUES (?, 'binary', ?, 'manual', ?)",
            [(i, body.value, body.session_id) for i in body.image_ids],
        )
    return {"labeled": len(body.image_ids)}


@app.post("/api/train_taste")
def train_taste():
    """Retrain the taste head on current labels and rescore everything.

    Seconds of CPU once embeddings exist (run app.embed for new images first)."""
    from . import taste

    try:
        return taste.train()
    except SystemExit as e:
        raise HTTPException(409, str(e))


@app.get("/api/tournament")
def tournament(size: int = 6):
    """A screen of liked images for best-of-N ranking; least-compared first."""
    with conn() as db:
        rows = db.execute(
            """SELECT i.id,
                      (SELECT count(*) FROM labels pw WHERE pw.kind = 'pairwise'
                       AND (pw.image_id = i.id OR pw.opponent_image_id = i.id)) AS comparisons
               FROM images i
               JOIN current_labels c ON c.image_id = i.id AND c.kind = 'binary' AND c.value = 1.0
               WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)
               ORDER BY comparisons, random() LIMIT ?""",
            (size,),
        ).fetchall()
    return {"items": [dict(r) for r in rows]}


class TournamentIn(BaseModel):
    winner_id: int
    loser_ids: list[int]
    session_id: int | None = None


@app.post("/api/tournament")
def tournament_vote(body: TournamentIn):
    """One click on the best of a screen = N-1 implied pairwise wins."""
    losers = [i for i in body.loser_ids if i != body.winner_id]
    if not losers:
        raise HTTPException(422, "need at least one loser")
    with conn() as db:
        db.executemany(
            "INSERT INTO labels (image_id, kind, value, opponent_image_id, source, session_id)"
            " VALUES (?, 'pairwise', 1, ?, 'manual', ?)",
            [(body.winner_id, loser, body.session_id) for loser in losers],
        )
    return {"pairs": len(losers)}


@app.get("/api/rankings")
def rankings(limit: int = 50):
    """Bradley-Terry strengths from pairwise votes (simple MM iteration)."""
    with conn() as db:
        pairs = db.execute(
            "SELECT image_id AS w, opponent_image_id AS l FROM labels WHERE kind = 'pairwise'"
        ).fetchall()
        if not pairs:
            return {"items": [], "pairs": 0}
        ids = sorted({p["w"] for p in pairs} | {p["l"] for p in pairs})
        idx = {v: k for k, v in enumerate(ids)}
        wins = [[0] * len(ids) for _ in ids]
        for p in pairs:
            wins[idx[p["w"]]][idx[p["l"]]] += 1
        strength = [1.0] * len(ids)
        for _ in range(50):  # MM algorithm (Hunter 2004)
            new = []
            for i in range(len(ids)):
                num = sum(wins[i][j] for j in range(len(ids)))
                den = sum(
                    (wins[i][j] + wins[j][i]) / (strength[i] + strength[j])
                    for j in range(len(ids)) if j != i and (wins[i][j] or wins[j][i])
                )
                new.append(num / den if den else strength[i])
            total = sum(new) or 1.0
            strength = [s * len(new) / total for s in new]
        ranked = sorted(zip(ids, strength), key=lambda t: -t[1])[:limit]
        return {"pairs": len(pairs),
                "items": [{"id": i, "strength": round(s, 3)} for i, s in ranked]}


@app.get("/api/model")
def model_info():
    with conn() as db:
        row = db.execute(
            "SELECT model, count(*) AS scored, max(created_at) AS at"
            " FROM predictions WHERE model LIKE 'taste:%' GROUP BY model"
            " ORDER BY model DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row and row["model"] else {"model": None}


@app.post("/api/label")
def add_label(body: LabelIn):
    if body.value not in VALID_BINARY:
        raise HTTPException(422, "value must be 1, 0.5 or 0")
    with conn() as db:
        if not db.execute("SELECT 1 FROM images WHERE id = ?", (body.image_id,)).fetchone():
            raise HTTPException(404, "unknown image")
        cur = db.execute(
            "INSERT INTO labels (image_id, kind, value, source, session_id)"
            " VALUES (?, 'binary', ?, 'manual', ?)",
            (body.image_id, body.value, body.session_id),
        )
        return {"label_id": cur.lastrowid}


@app.post("/api/exclude")
def exclude(body: ExcludeIn):
    with conn() as db:
        cur = db.execute(
            "INSERT INTO labels (image_id, kind, value, source, session_id)"
            " VALUES (?, 'exclude', 1, 'manual', ?)",
            (body.image_id, body.session_id),
        )
        return {"label_id": cur.lastrowid}


@app.post("/api/undo")
def undo(body: UndoIn):
    """Remove the most recent manual label (scoped to session when given)."""
    with conn() as db:
        if body.session_id is not None:
            row = db.execute(
                "SELECT id, image_id, kind, value FROM labels"
                " WHERE source = 'manual' AND session_id = ? ORDER BY id DESC LIMIT 1",
                (body.session_id,),
            ).fetchone()
        else:
            row = db.execute(
                "SELECT id, image_id, kind, value FROM labels"
                " WHERE source = 'manual' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            raise HTTPException(404, "nothing to undo")
        db.execute("DELETE FROM labels WHERE id = ?", (row["id"],))
        return {"image_id": row["image_id"], "kind": row["kind"], "value": row["value"]}


class IngestIn(BaseModel):
    data_b64: str                  # raw image bytes, base64
    image_url: str | None = None   # where the bytes came from
    page_url: str | None = None    # the gallery page
    value: float | None = None     # optional immediate rating
    session_id: int | None = None


@app.post("/api/ingest")
def ingest(body: IngestIn):
    """Ingest an image from the web (browser extension or URL importer).

    Content-addressed: re-ingesting known bytes only adds source rows, so the
    same image rated on two sites stays one record."""
    try:
        data = base64.b64decode(body.data_b64)
        info = inspect_image(data)
    except Exception:
        raise HTTPException(422, "not a decodable image")
    if body.value is not None and body.value not in VALID_BINARY:
        raise HTTPException(422, "value must be 1, 0.5 or 0")

    sha = hashlib.sha256(data).hexdigest()
    ext = (info["format"] or "png").lower()
    with conn() as db:
        row = db.execute("SELECT id FROM images WHERE sha256 = ?", (sha,)).fetchone()
        if row:
            image_id, created = row["id"], False
        else:
            store = DATA_DIR / "ingested"
            store.mkdir(parents=True, exist_ok=True)
            path = store / f"{sha[:16]}.{ext}"
            path.write_bytes(data)
            cur = db.execute(
                """INSERT INTO images (sha256, width, height, format, file_size,
                                       prompt, negative_prompt, model_hash, seed, gen_params_raw)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sha, info["width"], info["height"], info["format"], len(data),
                 info.get("prompt"), info.get("negative_prompt"),
                 info.get("model_hash"), info.get("seed"), info.get("gen_params_raw")),
            )
            image_id, created = cur.lastrowid, True
            db.execute(
                "INSERT INTO image_sources (image_id, kind, location) VALUES (?, 'local', ?)",
                (image_id, str(path.resolve())),
            )
        for url in {u for u in (body.image_url, body.page_url) if u}:
            db.execute(
                """INSERT INTO image_sources (image_id, kind, location) VALUES (?, 'url', ?)
                   ON CONFLICT (image_id, location) DO UPDATE SET last_verified = datetime('now')""",
                (image_id, url),
            )
        label_id = None
        if body.value is not None:
            cur = db.execute(
                "INSERT INTO labels (image_id, kind, value, source, session_id)"
                " VALUES (?, 'binary', ?, 'manual', ?)",
                (image_id, body.value, body.session_id),
            )
            label_id = cur.lastrowid
    return {"image_id": image_id, "created": created, "label_id": label_id}


@app.post("/api/sessions")
def start_session(body: SessionIn):
    with conn() as db:
        cur = db.execute("INSERT INTO sessions (name) VALUES (?)", (body.name,))
        return {"session_id": cur.lastrowid}


@app.get("/api/stats")
def stats(session_id: int | None = None):
    with conn() as db:
        total = db.execute("SELECT count(*) AS n FROM images").fetchone()["n"]
        by_value = {
            r["value"]: r["n"]
            for r in db.execute(
                "SELECT value, count(*) AS n FROM current_labels"
                " WHERE kind = 'binary' GROUP BY value"
            )
        }
        excluded = db.execute(
            "SELECT count(*) AS n FROM current_labels WHERE kind = 'exclude'"
        ).fetchone()["n"]
        labeled = sum(by_value.values())
        out = {
            "total": total,
            "liked": by_value.get(1.0, 0),
            "maybe": by_value.get(0.5, 0),
            "disliked": by_value.get(0.0, 0),
            "excluded": excluded,
            "unlabeled": total - labeled - excluded,
        }
        if session_id is not None:
            row = db.execute(
                """SELECT count(*) AS n,
                          (julianday('now') - julianday(min(created_at))) * 24 * 60 AS minutes
                   FROM labels WHERE session_id = ? AND source = 'manual'""",
                (session_id,),
            ).fetchone()
            out["session_count"] = row["n"]
            out["session_per_min"] = round(row["n"] / row["minutes"], 1) if row["minutes"] else None
        return out


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
