"""imghotornot v2 — local FastAPI server: rating API + static UI + image serving."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import get_conn

app = FastAPI(title="imghotornot")

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


@app.get("/api/queue")
def queue(limit: int = 20):
    """Unlabeled, non-excluded images. Phase 3 swaps this for the uncertainty queue."""
    with conn() as db:
        rows = db.execute(
            """SELECT i.id, i.width, i.height, i.prompt, i.model_hash
               FROM images i
               WHERE NOT EXISTS (SELECT 1 FROM current_labels c
                                 WHERE c.image_id = i.id AND c.kind IN ('binary','exclude'))
               ORDER BY i.id
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return {"items": [dict(r) for r in rows]}


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
