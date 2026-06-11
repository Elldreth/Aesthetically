"""Aesthetically — local FastAPI server: rating API + static UI + image serving."""
from __future__ import annotations

import base64
import logging
import os
import secrets
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import DATA_DIR, get_conn
from .ingest import DECODE_ERRORS, register_bytes

log = logging.getLogger("aesthetically")

# Loopback app, hostile internet: any website the user visits can fire POSTs at
# 127.0.0.1 (text/plain bodies skip CORS preflight), and DNS rebinding defeats
# origin checks. Defense: (1) strict Host allowlist, (2) a per-install token
# required on every mutating request. The token is set as a SameSite=Strict
# cookie when the UI loads; app.js echoes it in the X-Aesth-Token header.
# The extension stores it via its options page.
ALLOWED_HOSTS = {h.strip() for h in os.environ.get(
    "AESTH_ALLOWED_HOSTS", "127.0.0.1:8787,localhost:8787").split(",")}
TOKEN_PATH = DATA_DIR / "token.txt"


def _load_token() -> str:
    if TOKEN_PATH.is_file():
        return TOKEN_PATH.read_text(encoding="utf-8").strip()
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    TOKEN_PATH.write_text(token, encoding="utf-8")
    return token


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_conn().close()          # apply schema once, eagerly
    _app.state.token = _load_token()
    yield


app = FastAPI(title="Aesthetically", lifespan=lifespan)

# the browser extension calls /api/ingest from chrome-extension:// origins
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome|moz)-extension://.*$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Aesth-Token"],
)


@app.middleware("http")
async def guard(request: Request, call_next):
    host = request.headers.get("host", "")
    if host not in ALLOWED_HOSTS:
        return JSONResponse({"detail": "unknown host"}, status_code=421)
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path.startswith("/api/"):
        token = getattr(request.app.state, "token", None)
        sent = request.headers.get("x-aesth-token") or request.cookies.get("aesth_token")
        if token and not (sent and secrets.compare_digest(sent, token)):
            return JSONResponse({"detail": "missing or invalid token"}, status_code=403)
    response = await call_next(request)
    return response

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


@contextmanager
def conn():
    """Per-request connection: commits on success, always closed."""
    db = get_conn()
    try:
        with db:
            yield db
    finally:
        db.close()


def _ui_response(page: str) -> Response:
    resp = FileResponse(STATIC_DIR / page)
    token = getattr(app.state, "token", None)
    if token:
        resp.set_cookie("aesth_token", token, samesite="strict", httponly=False,
                        max_age=365 * 24 * 3600)
    return resp


@app.get("/")
def index():
    return _ui_response("index.html")


@app.get("/static/{page}.html")
def ui_page(page: str):
    safe = Path(page).name + ".html"
    if not (STATIC_DIR / safe).is_file():
        raise HTTPException(404)
    return _ui_response(safe)


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


# scalar subquery (not a JOIN): an image briefly holding two taste:% rows during
# a retrain race must not appear twice in the queue
_UNRATED = """
    FROM (SELECT i.*,
                 (SELECT score FROM predictions p WHERE p.image_id = i.id
                  AND p.model LIKE 'taste:%' ORDER BY p.created_at DESC, p.model DESC
                  LIMIT 1) AS score
          FROM images i) i
    WHERE NOT EXISTS (SELECT 1 FROM current_labels c
                      WHERE c.image_id = i.id AND c.kind IN ('binary','exclude'))
      AND NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)
"""
_QUEUE_ORDER = {
    "default": "i.id",
    "uncertain": "CASE WHEN i.score IS NULL THEN 1 ELSE 0 END, ABS(i.score - 0.5)",
    "best": "i.score IS NULL, i.score DESC",
    "worst": "i.score IS NULL, i.score ASC",
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
            f"""SELECT i.id, i.width, i.height, i.prompt, i.model_hash, i.score
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
        tmp_path = thumb_dir / f".{image_id}.{os.getpid()}.tmp"
        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                img.thumbnail((320, 320))
                img.save(tmp_path, "WEBP", quality=80)
            os.replace(tmp_path, thumb_path)  # atomic: concurrent firsts can't collide
        except DECODE_ERRORS:
            raise HTTPException(415, "source image is unreadable")
        finally:
            tmp_path.unlink(missing_ok=True)
    return FileResponse(thumb_path, media_type="image/webp")


_SCORED_IMAGES = """
    FROM (SELECT i.*,
                 (SELECT score FROM predictions p WHERE p.image_id = i.id
                  AND p.model LIKE 'taste:%' ORDER BY p.created_at DESC, p.model DESC
                  LIMIT 1) AS score
          FROM images i) i
"""
_GRID_FILTER = {
    "unrated": _UNRATED,
    "liked": _SCORED_IMAGES + " JOIN current_labels c ON c.image_id = i.id"
             " AND c.kind = 'binary' AND c.value = 1.0",
    "maybe": _SCORED_IMAGES + " JOIN current_labels c ON c.image_id = i.id"
             " AND c.kind = 'binary' AND c.value = 0.5",
    "disliked": _SCORED_IMAGES + " JOIN current_labels c ON c.image_id = i.id"
                " AND c.kind = 'binary' AND c.value = 0.0",
    "all": _SCORED_IMAGES + " WHERE 1=1",
}
_GRID_ORDER = dict(_QUEUE_ORDER, newest="i.id DESC")


@app.get("/api/grid")
def grid(mode: str = "worst", limit: int = 60, offset: int = 0,
         filter: str = "unrated"):
    """Paged images for the grid, sorted by taste score. filter selects the
    label bucket; rated images stay browsable (newest mode shows recent work)."""
    order = _GRID_ORDER.get(mode)
    source = _GRID_FILTER.get(filter)
    if order is None:
        raise HTTPException(422, f"mode must be one of {sorted(_GRID_ORDER)}")
    if source is None:
        raise HTTPException(422, f"filter must be one of {sorted(_GRID_FILTER)}")
    with conn() as db:
        total = db.execute(f"SELECT count(*) AS n {source}").fetchone()["n"]
        rows = db.execute(
            f"""SELECT i.id, i.score {source} ORDER BY {order} LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
    return {"total": total, "items": [dict(r) for r in rows]}


class BulkLabelIn(BaseModel):
    image_ids: list[int]
    value: float
    session_id: int | None = None


def _require_images(db, image_ids: list[int]) -> None:
    placeholders = ",".join("?" * len(image_ids))
    n = db.execute(
        f"SELECT count(*) AS n FROM images WHERE id IN ({placeholders})",
        image_ids,
    ).fetchone()["n"]
    if n != len(set(image_ids)):
        raise HTTPException(404, "unknown image id in request")


@app.post("/api/labels/bulk")
def bulk_label(body: BulkLabelIn):
    if body.value not in VALID_BINARY:
        raise HTTPException(422, "value must be 1, 0.5 or 0")
    if not body.image_ids:
        return {"labeled": 0}
    if len(body.image_ids) > 500:
        raise HTTPException(422, "too many images in one request")
    with conn() as db:
        _require_images(db, body.image_ids)
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


# ---- studio: the Artifex closed loop ----

def _studio_guard(e: Exception) -> HTTPException:
    """Long stack traces and internal paths stay in the server log."""
    log.exception("studio operation failed")
    return HTTPException(502, "operation failed — check the server log")


class BestOfNIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    n: int = Field(default=4, ge=1, le=8)
    model: str | None = None
    size: str = "832x1216"
    loras: list[dict] | None = None


@app.post("/api/studio/best_of_n")
def studio_best_of_n(body: BestOfNIn):
    from . import studio

    try:
        return {"items": studio.best_of_n(body.prompt, body.n, body.model,
                                          body.size, body.loras)}
    except Exception as e:
        raise _studio_guard(e)


class TrainLoraIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    k: int = Field(default=40, ge=10, le=200)
    steps: int = Field(default=1200, ge=50, le=4000)
    rank: int = Field(default=16, ge=4, le=64)
    lr: float = Field(default=1e-4, gt=0, le=1e-2)


@app.post("/api/studio/train_lora")
def studio_train_lora(body: TrainLoraIn):
    from . import studio

    try:
        return studio.train_taste_lora(body.name, body.k, body.steps, body.rank, body.lr)
    except Exception as e:
        raise _studio_guard(e)


@app.get("/api/studio/runs")
def studio_runs():
    with conn() as db:
        rows = db.execute(
            "SELECT id, name, status, started_at, finished_at, artifact_path,"
            " dataset_fingerprint_json FROM training_runs ORDER BY id DESC LIMIT 50"
        ).fetchall()
    return {"items": [dict(r) for r in rows]}


@app.get("/api/studio/runs/{run_id}")
def studio_run_status(run_id: int):
    from . import studio

    try:
        return studio.poll_run(run_id)
    except Exception as e:
        raise _studio_guard(e)


class EvalLoraIn(BaseModel):
    run_id: int
    lora_name: str | None = None
    prompts: list[str] | None = Field(default=None, max_length=12)
    seeds_per_prompt: int = Field(default=2, ge=1, le=4)
    model: str | None = None


@app.post("/api/studio/eval_lora")
def studio_eval_lora(body: EvalLoraIn):
    from . import studio

    try:
        return studio.eval_lora(body.run_id, body.lora_name, body.prompts,
                                body.seeds_per_prompt, body.model)
    except Exception as e:
        raise _studio_guard(e)


@app.get("/api/studio/health")
def studio_health():
    from .artifex_client import ArtifexClient

    with ArtifexClient(timeout=3.0) as client:
        up = client.is_up()
        out = {"artifex": up}
        if up:
            try:
                import httpx

                r = httpx.get(client.base_url + "/health", timeout=5.0).json()
                out.update({k: r.get(k) for k in ("models", "vram") if k in r})
                r2 = httpx.get(client.base_url + "/v1/loras", timeout=5.0).json()
                out["loras"] = [l.get("name") for l in r2] if isinstance(r2, list) else r2
            except Exception:
                pass
    return out


@app.get("/api/studio/probe_prompts")
def studio_probe_prompts():
    from . import studio

    return {"prompts": studio.default_probe_prompts()}


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
        _require_images(db, [body.image_id])
        cur = db.execute(
            "INSERT INTO labels (image_id, kind, value, source, session_id)"
            " VALUES (?, 'exclude', 1, 'manual', ?)",
            (body.image_id, body.session_id),
        )
        return {"label_id": cur.lastrowid}


@app.post("/api/undo")
def undo(body: UndoIn):
    """Remove the most recent manual label in the given session.

    session_id is required: a global undo would let one client erase another's
    most recent rating."""
    if body.session_id is None:
        raise HTTPException(422, "session_id is required")
    with conn() as db:
        row = db.execute(
            "SELECT id, image_id, kind, value FROM labels"
            " WHERE source = 'manual' AND session_id = ? ORDER BY id DESC LIMIT 1",
            (body.session_id,),
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
    if len(body.data_b64) > 45_000_000:  # ~33MB binary
        raise HTTPException(413, "image too large")
    try:
        data = base64.b64decode(body.data_b64)
    except Exception:
        raise HTTPException(422, "not decodable base64")
    if body.value is not None and body.value not in VALID_BINARY:
        raise HTTPException(422, "value must be 1, 0.5 or 0")

    with conn() as db:
        try:
            image_id, created = register_bytes(
                db, data, store_dir=DATA_DIR / "ingested",
                image_url=body.image_url, page_url=body.page_url,
            )
        except DECODE_ERRORS:
            raise HTTPException(422, "not a decodable image")
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
        # distinct images with ANY current label — an image that is both rated
        # and excluded must not be subtracted twice
        labeled_any = db.execute(
            "SELECT count(DISTINCT image_id) AS n FROM current_labels"
            " WHERE kind IN ('binary','exclude')"
        ).fetchone()["n"]
        out = {
            "total": total,
            "liked": by_value.get(1.0, 0),
            "maybe": by_value.get(0.5, 0),
            "disliked": by_value.get(0.0, 0),
            "excluded": excluded,
            "unlabeled": total - labeled_any,
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
