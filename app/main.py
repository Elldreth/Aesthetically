"""Aesthetically — local FastAPI server: rating API + static UI + image serving."""
from __future__ import annotations

import base64
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import DATA_DIR, conn, get_conn
from .ingest import DECODE_ERRORS, register_bytes
from .jobs import manager as jobs

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
    # the UI assets change often (single-user local app) — make the browser
    # revalidate so a stale app.js/app.css never lingers after an update
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

STATIC_DIR = Path(__file__).resolve().parent / "static"

VALID_BINARY = {0.0, 0.5, 1.0}


_NOT_A_FOLDER_MSG = ("not a folder the server can see — check the path "
                     "(remove surrounding quotes; mapped drives must be visible "
                     "to the server)")


def _clean_path(raw: str) -> Path:
    """Tolerate paths pasted with surrounding quotes (Windows 'Copy as path'
    wraps in double quotes) or stray whitespace."""
    return Path(raw.strip().strip('"').strip("'").strip())


def _thumb_for(src: str) -> Path:
    """Cached 320px WebP thumbnail for a source file, keyed by path+mtime+size.

    Shared by /thumb and the scan grid so multi-megabyte originals are never
    shipped to the browser to be squeezed into a grid cell. draft() lets the
    JPEG decoder load big files at a reduced scale — fast and low-memory.
    Raises DECODE_ERRORS on an unreadable source.
    """
    import hashlib

    from PIL import Image

    st = os.stat(src)
    key = hashlib.sha1(
        f"{os.path.abspath(src)}|{st.st_mtime_ns}|{st.st_size}".encode()
    ).hexdigest()[:20]
    thumb_dir = DATA_DIR / "thumbs"
    thumb_path = thumb_dir / f"{key}.webp"
    if thumb_path.is_file():
        return thumb_path
    thumb_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = thumb_dir / f".{key}.{os.getpid()}.tmp"
    try:
        with Image.open(src) as img:
            img.draft("RGB", (640, 640))      # reduced-scale JPEG decode (no-op for PNG)
            img = img.convert("RGB")
            img.thumbnail((320, 320))
            img.save(tmp_path, "WEBP", quality=80)
        os.replace(tmp_path, thumb_path)       # atomic: concurrent firsts can't collide
    finally:
        tmp_path.unlink(missing_ok=True)
    return thumb_path


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


def _ui_response(page: str) -> Response:
    resp = FileResponse(STATIC_DIR / page)
    token = getattr(app.state, "token", None)
    if token:
        resp.set_cookie("aesth_token", token, samesite="strict", httponly=False,
                        max_age=365 * 24 * 3600)
    return resp


@app.get("/")
def index():
    return _ui_response("home.html")


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
                  LIMIT 1) AS score,
                 (SELECT score FROM hand_scores h WHERE h.image_id = i.id) AS hand
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
    with conn() as db:
        rows = db.execute(
            "SELECT location FROM image_sources WHERE image_id = ? AND kind = 'local'",
            (image_id,),
        ).fetchall()
    src = next((r["location"] for r in rows if os.path.isfile(r["location"])), None)
    if not src:
        raise HTTPException(404, "no readable local source for image")
    try:
        return FileResponse(_thumb_for(src), media_type="image/webp")
    except DECODE_ERRORS:
        raise HTTPException(415, "source image is unreadable")


_SCORED_IMAGES = """
    FROM (SELECT i.*,
                 (SELECT score FROM predictions p WHERE p.image_id = i.id
                  AND p.model LIKE 'taste:%' ORDER BY p.created_at DESC, p.model DESC
                  LIMIT 1) AS score,
                 (SELECT score FROM hand_scores h WHERE h.image_id = i.id) AS hand
          FROM images i) i
"""
# every fragment ends with a WHERE so a style clause can append uniformly
_LABELED = (_SCORED_IMAGES + " JOIN current_labels c ON c.image_id = i.id"
            " WHERE c.kind = 'binary' AND c.value = ")
_GRID_FILTER = {
    "unrated": _UNRATED,
    "liked": _LABELED + "1.0",
    "maybe": _LABELED + "0.5",
    "disliked": _LABELED + "0.0",
    "removed": _SCORED_IMAGES + " JOIN current_labels c ON c.image_id = i.id"
               " WHERE c.kind = 'exclude'",
    "bad_hands": _SCORED_IMAGES + " WHERE i.hand IS NOT NULL AND i.hand < 0.5"
                 " AND NOT EXISTS (SELECT 1 FROM current_labels c"
                 " WHERE c.image_id = i.id AND c.kind = 'exclude')",
    "all": _SCORED_IMAGES + " WHERE 1=1",
}
# worst_hands: lowest hand-score first (probable bad hands), unscored images last
_GRID_ORDER = dict(_QUEUE_ORDER, newest="i.id DESC",
                   worst_hands="i.hand IS NULL, i.hand ASC")
_STYLE_CLAUSE = {
    "anime": " AND EXISTS (SELECT 1 FROM image_styles st WHERE st.image_id=i.id AND st.style='anime')",
    "realistic": " AND EXISTS (SELECT 1 FROM image_styles st WHERE st.image_id=i.id AND st.style='realistic')",
    "untagged": " AND NOT EXISTS (SELECT 1 FROM image_styles st WHERE st.image_id=i.id)",
}


@app.get("/api/grid")
def grid(mode: str = "worst", limit: int = 60, offset: int = 0,
         filter: str = "unrated", style: str | None = None):
    """Paged images for the grid, sorted by taste score. filter selects the
    label bucket; style (anime/realistic/untagged) optionally narrows further."""
    order = _GRID_ORDER.get(mode)
    source = _GRID_FILTER.get(filter)
    if order is None:
        raise HTTPException(422, f"mode must be one of {sorted(_GRID_ORDER)}")
    if source is None:
        raise HTTPException(422, f"filter must be one of {sorted(_GRID_FILTER)}")
    if style:
        clause = _STYLE_CLAUSE.get(style)
        if clause is None:
            raise HTTPException(422, f"style must be one of {sorted(_STYLE_CLAUSE)}")
        source = source + clause
    with conn() as db:
        total = db.execute(f"SELECT count(*) AS n {source}").fetchone()["n"]
        rows = db.execute(
            f"""SELECT i.id, i.score, i.hand {source} ORDER BY {order} LIMIT ? OFFSET ?""",
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


class BulkIdsIn(BaseModel):
    image_ids: list[int] = Field(min_length=1, max_length=2000)
    session_id: int | None = None


@app.post("/api/exclude/bulk")
def bulk_exclude(body: BulkIdsIn):
    """Remove images (depth maps, junk) — excludes them from the queue and from
    training. Reversible via /api/exclude/restore."""
    with conn() as db:
        _require_images(db, body.image_ids)
        db.executemany(
            "INSERT INTO labels (image_id, kind, value, source, session_id)"
            " VALUES (?, 'exclude', 1, 'manual', ?)",
            [(i, body.session_id) for i in body.image_ids],
        )
    return {"excluded": len(body.image_ids)}


@app.post("/api/exclude/restore")
def bulk_restore(body: BulkIdsIn):
    """Un-remove: drop the exclude labels so the images return to the pool."""
    with conn() as db:
        placeholders = ",".join("?" * len(body.image_ids))
        cur = db.execute(
            f"DELETE FROM labels WHERE kind='exclude' AND image_id IN ({placeholders})",
            body.image_ids,
        )
    return {"restored": cur.rowcount}


@app.post("/api/train_taste")
def train_taste():
    """Retrain a taste model per style (anime, realistic) on current labels and
    rescore each style's images. Seconds of CPU once embeddings exist."""
    from . import taste

    results = taste.train_styles()
    if all("skipped" in r for r in results.values()):
        raise HTTPException(409, "; ".join(r.get("skipped", "") for r in results.values()))
    return results


@app.get("/api/tournament")
def tournament(size: int = 6, style: str | None = None, exclude: str | None = None):
    """A screen of liked images of ONE style for best-of-N ranking, least-
    compared first. Ranking across styles is apples-vs-oranges, so a screen is
    always one style. With style omitted, defaults to whichever style has the
    most liked images. exclude = comma-separated ids to skip (e.g. images
    already on screen when fetching a replacement)."""
    try:
        skip = [int(x) for x in exclude.split(",") if x.strip()] if exclude else []
    except ValueError:
        raise HTTPException(422, "exclude must be comma-separated ids")
    with conn() as db:
        if style is None:
            row = db.execute(
                "SELECT s.style, count(*) AS n FROM current_labels c"
                " JOIN image_styles s ON s.image_id = c.image_id"
                " WHERE c.kind='binary' AND c.value=1.0 GROUP BY s.style"
                " ORDER BY n DESC LIMIT 1").fetchone()
            style = row["style"] if row else "anime"
        if style not in ("anime", "realistic"):
            raise HTTPException(422, "style must be anime or realistic")
        not_in = f"AND i.id NOT IN ({','.join('?' * len(skip))})" if skip else ""
        rows = db.execute(
            f"""SELECT i.id,
                      (SELECT count(*) FROM labels pw WHERE pw.kind = 'pairwise'
                       AND (pw.image_id = i.id OR pw.opponent_image_id = i.id)) AS comparisons
               FROM images i
               JOIN current_labels c ON c.image_id = i.id AND c.kind = 'binary' AND c.value = 1.0
               JOIN image_styles st ON st.image_id = i.id AND st.style = ?
               WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)
                 AND NOT EXISTS (SELECT 1 FROM current_labels e
                                 WHERE e.image_id = i.id AND e.kind = 'exclude')
                 {not_in}
               ORDER BY comparisons, random() LIMIT ?""",
            (style, *skip, size),
        ).fetchall()
    return {"style": style, "items": [dict(r) for r in rows]}


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
    """RuntimeError carries a user-actionable message (Artifex busy, too few
    images, no model) — surface it. Everything else is masked; details go to
    the server log."""
    if isinstance(e, RuntimeError):
        return HTTPException(409, str(e))
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


@app.get("/api/studio/presets")
def studio_presets():
    """Training presets + clamp + per-style dataset sizes + the style→checkpoint
    map, so the UI shows the exact steps/lr/rank math and which base model fits."""
    from . import studio

    return {"presets": studio.LORA_PRESETS, "default": studio.DEFAULT_PRESET,
            "default_max_images": studio.DEFAULT_MAX_IMAGES,
            "steps_min": studio.STEPS_MIN, "steps_max": studio.STEPS_MAX,
            "checkpoints": studio.STYLE_CHECKPOINTS,
            "liked_counts": studio.liked_counts()}


class TrainLoraIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    max_images: int = Field(default=60, ge=10, le=200)
    preset: str = Field(default="balanced", pattern="^(subtle|balanced|strong)$")
    style: str | None = Field(default=None, pattern="^(anime|realistic|all)$")
    model: str | None = None
    steps: int | None = Field(default=None, ge=200, le=6000)  # optional override
    lr: float | None = Field(default=None, gt=0, le=1e-2)
    rank: int | None = Field(default=None, ge=4, le=128)


@app.post("/api/studio/train_lora")
def studio_train_lora(body: TrainLoraIn):
    from . import studio

    style = None if body.style in (None, "all") else body.style
    try:
        return studio.train_taste_lora(body.name, max_images=body.max_images,
                                       preset=body.preset, model=body.model,
                                       style=style, steps=body.steps,
                                       lr=body.lr, rank=body.rank)
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


@app.get("/api/studio/clusters")
def studio_clusters(min_cluster_size: int = 10):
    """Coherent style clusters among the liked images (for per-cluster LoRAs)."""
    from .cluster import cluster_likes

    try:
        clusters = cluster_likes(min_cluster_size=min_cluster_size)
    except Exception as e:
        raise _studio_guard(e)
    return {"clusters": [{k: c[k] for k in
                          ("cluster", "size", "cohesion", "samples", "image_ids")}
                         for c in clusters]}


class TrainClusterIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    image_ids: list[int] = Field(min_length=10, max_length=2000)
    preset: str = Field(default="balanced", pattern="^(subtle|balanced|strong)$")
    model: str | None = None
    max_images: int = Field(default=60, ge=10, le=200)
    steps: int | None = Field(default=None, ge=200, le=6000)
    lr: float | None = Field(default=None, gt=0, le=1e-2)
    rank: int | None = Field(default=None, ge=4, le=128)


@app.post("/api/studio/train_cluster_lora")
def studio_train_cluster_lora(body: TrainClusterIn):
    from . import studio

    try:
        return studio.submit_lora(body.name, body.image_ids, preset=body.preset,
                                  model=body.model, max_images=body.max_images,
                                  steps=body.steps, lr=body.lr, rank=body.rank)
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


# ---- unified background job queue ----


@app.get("/api/jobs")
def jobs_list():
    """All jobs (queued/running/recent) for the header status indicator."""
    return {"items": jobs.list_jobs(), "active": jobs.active_count()}


@app.get("/api/jobs/{job_id}")
def jobs_get(job_id: int):
    """One job's status + result, for clients polling a specific submission."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.as_dict()


@app.post("/api/jobs/{job_id}/cancel")
def jobs_cancel(job_id: int):
    return {"cancelled": jobs.cancel(job_id)}


class DedupeIn(BaseModel):
    # 0 = identical perceptual hash (true duplicates); higher = fuzzier.
    phash_dist: int = Field(default=0, ge=0, le=8)


@app.post("/api/dedupe")
def dedupe_remove(body: DedupeIn):
    """Find near-identical images and remove all but one per group (reversible
    exclude; nothing deleted from disk). Runs as a background job."""
    from . import dedupe

    job = jobs.submit(
        "dedupe", "removing duplicates",
        lambda progress, cancel: dedupe.remove_duplicates(progress, cancel, body.phash_dist),
    )
    return {"started": True, "job_id": job.id}


class IngestFolderIn(BaseModel):
    path: str = Field(min_length=1, max_length=500)
    recursive: bool = True


@app.post("/api/ingest_folder")
def ingest_folder(body: IngestFolderIn):
    """Register a folder (read-only), then hash/embed/score — queued as a job."""
    from .ingest import run_folder_ingest

    folder = _clean_path(body.path)
    if not folder.is_dir():
        raise HTTPException(422, _NOT_A_FOLDER_MSG)
    job = jobs.submit(
        "ingest", str(folder),
        lambda progress, cancel: run_folder_ingest(
            folder, progress, recursive=body.recursive, cancel=cancel),
    )
    return {"started": True, "job_id": job.id}


class SelectIn(BaseModel):
    out: str = Field(min_length=1, max_length=500)
    top: int | None = Field(default=None, ge=1, le=100_000)
    min_score: float | None = Field(default=None, ge=0, le=1)
    buckets: bool = False
    unlabeled_only: bool = False
    mode: str = Field(default="copy", pattern="^(copy|link|move)$")


@app.post("/api/select")
def select_images(body: SelectIn):
    """Export top-predicted images from the collection to a folder — queued."""
    from .select import run_select

    if not (body.top or body.min_score is not None or body.buckets):
        raise HTTPException(422, "pick top, min_score, or buckets")
    out = _clean_path(body.out)
    if out.exists() and not out.is_dir():
        raise HTTPException(422, "out exists and is not a folder")
    job = jobs.submit(
        "export", str(out),
        lambda progress, cancel: run_select(
            out, top=body.top, min_score=body.min_score, buckets=body.buckets,
            unlabeled_only=body.unlabeled_only, mode=body.mode, progress=progress),
    )
    return {"started": True, "job_id": job.id}


# ---- folder scoring (persisted, kept out of the training collection) ----

# Each scan is a row in `scans`; results live in `scan_items` and survive
# restarts. Result thumbnails are served as /scan/img/{scan_id}/{rank} where
# rank is the position in score-desc order — a stable, persistent address, so
# a lingering page can never show a different scan's image.

_JOB_STATE_TO_SCAN = {"queued": "starting", "running": "scoring",
                      "done": "done", "failed": "failed", "cancelled": "done"}


def _latest_scan_id(db) -> int | None:
    row = db.execute("SELECT max(id) AS m FROM scans WHERE count IS NOT NULL").fetchone()
    return row["m"]


class ScanIn(BaseModel):
    path: str = Field(min_length=1, max_length=500)


@app.post("/api/scan")
def scan_folder(body: ScanIn):
    """Score a folder WITHOUT adding it to the collection. Persisted; queued."""
    from .scan import MODEL_NAME, run_scan
    from .scorer import latest_head

    folder = _clean_path(body.path)
    if not folder.is_dir():
        raise HTTPException(422, _NOT_A_FOLDER_MSG)
    head = latest_head()
    if head is None:
        raise HTTPException(409, "no taste model yet — rate and train first")
    with conn() as db:
        cur = db.execute(
            "INSERT INTO scans (path, taste_model, embed_model) VALUES (?, ?, ?)",
            (str(folder), head["name"], MODEL_NAME),
        )
        scan_id = cur.lastrowid
    jobs.submit(
        "scan", str(folder),
        lambda progress, cancel: run_scan(folder, scan_id, progress, cancel=cancel),
    )
    return {"started": True, "scan_id": scan_id}


@app.get("/api/scan/status")
def scan_status():
    """Shape kept for scan.html: maps the latest scan job to state/scan_id/…"""
    job = jobs.latest("scan")
    if job is None:
        return {"state": "idle", "scan_id": None}
    p = job.progress
    return {"state": _JOB_STATE_TO_SCAN.get(job.state, job.state),
            "scan_id": p.get("scan_id"), "path": job.label,
            "done": p.get("done", 0), "total": p.get("total", 0),
            "model": p.get("model")}


@app.get("/api/scans")
def scans_list():
    """Past scans, newest first — for the 'reopen a scan' picker."""
    with conn() as db:
        rows = db.execute(
            "SELECT id, path, count, taste_model, created_at FROM scans"
            " WHERE count IS NOT NULL ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return {"items": [dict(r) for r in rows]}


@app.get("/api/scan/results")
def scan_results(scan_id: int | None = None, limit: int = 60, offset: int = 0):
    with conn() as db:
        sid = scan_id if scan_id is not None else _latest_scan_id(db)
        if sid is None:
            return {"total": 0, "scan_id": None, "path": None, "model": None, "items": []}
        meta = db.execute("SELECT path, taste_model, count FROM scans WHERE id = ?",
                          (sid,)).fetchone()
        if meta is None:
            raise HTTPException(404, "no such scan")
        rows = db.execute(
            "SELECT path, score FROM scan_items WHERE scan_id = ?"
            " ORDER BY score DESC, path LIMIT ? OFFSET ?",
            (sid, limit, offset),
        ).fetchall()
    return {"total": meta["count"] or 0, "scan_id": sid, "path": meta["path"],
            "model": meta["taste_model"],
            "items": [{"i": offset + j, "score": r["score"], "name": Path(r["path"]).name}
                      for j, r in enumerate(rows)]}


@app.get("/scan/img/{scan_id}/{rank}")
def scan_image(scan_id: int, rank: int, full: bool = False):
    """The rank-th image (score desc) in a persisted scan — a 320px thumbnail
    by default, or the full-resolution original with ?full=1 (for the lightbox)."""
    with conn() as db:
        row = db.execute(
            "SELECT path FROM scan_items WHERE scan_id = ?"
            " ORDER BY score DESC, path LIMIT 1 OFFSET ?",
            (scan_id, rank),
        ).fetchone()
    if row is None or not os.path.isfile(row["path"]):
        raise HTTPException(404)
    if full:
        return FileResponse(row["path"])
    try:
        return FileResponse(_thumb_for(row["path"]), media_type="image/webp")
    except DECODE_ERRORS:
        raise HTTPException(415, "source image is unreadable")


class ScanExportIn(BaseModel):
    out: str = Field(min_length=1, max_length=500)
    scan_id: int | None = None
    top: int = Field(default=50, ge=1, le=100_000)
    mode: str = Field(default="copy", pattern="^(copy|link|move)$")


def _export_files(rows: list, out: Path, mode: str, progress: dict) -> dict:
    """Copy/link/move scored files into out — runs as a background job."""
    import shutil

    out.mkdir(parents=True, exist_ok=True)
    progress.update(phase="exporting", total=len(rows), done=0)
    n = 0
    for i, r in enumerate(rows, 1):
        src = Path(r["path"])
        progress["done"] = i
        if not src.is_file():
            continue
        dst = out / f"{r['score']:.3f}_{src.name}"
        if dst.exists():
            continue
        if mode == "move":
            shutil.move(src, dst)
        elif mode == "link":
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        n += 1
    return {"count": n, "out": str(out)}


@app.post("/api/scan/export")
def scan_export(body: ScanExportIn):
    with conn() as db:
        sid = body.scan_id if body.scan_id is not None else _latest_scan_id(db)
        if sid is None:
            raise HTTPException(409, "no completed scan to export from")
        rows = db.execute(
            "SELECT path, score FROM scan_items WHERE scan_id = ?"
            " ORDER BY score DESC, path LIMIT ?",
            (sid, body.top),
        ).fetchall()
    out = _clean_path(body.out)
    job = jobs.submit(
        "export", str(out),
        lambda progress, cancel: _export_files([dict(r) for r in rows], out, body.mode, progress),
    )
    return {"started": True, "job_id": job.id, "scan_id": sid}


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


@app.get("/api/dashboard")
def dashboard():
    """Everything the Home page needs: collection counts, model status, and a
    single state-driven 'next step' so the workflow is legible."""
    from .scorer import latest_head

    with conn() as db:
        total = db.execute("SELECT count(*) AS n FROM images").fetchone()["n"]
        by_value = {r["value"]: r["n"] for r in db.execute(
            "SELECT value, count(*) AS n FROM current_labels WHERE kind='binary' GROUP BY value")}
        excluded = db.execute(
            "SELECT count(*) AS n FROM current_labels WHERE kind='exclude'").fetchone()["n"]
        labeled_any = db.execute(
            "SELECT count(DISTINCT image_id) AS n FROM current_labels"
            " WHERE kind IN ('binary','exclude')").fetchone()["n"]
        n_sources = db.execute(
            "SELECT count(DISTINCT location) AS n FROM image_sources WHERE kind='local'"
        ).fetchone()["n"]
        styles = {r["style"]: r["n"] for r in db.execute(
            "SELECT style, count(*) AS n FROM image_styles GROUP BY style")}
        # one model per style
        models = {}
        for st in ("anime", "realistic"):
            h = latest_head(st)
            if not h:
                continue
            m = h.get("metrics", {})
            ls = db.execute(
                "SELECT count(*) AS n FROM labels l"
                " JOIN image_styles s ON s.image_id = l.image_id AND s.style = ?"
                " WHERE l.source='manual' AND l.kind='binary' AND l.created_at > ?",
                (st, h["trained_at"])).fetchone()["n"]
            models[st] = {"name": h["name"], "trained_at": h["trained_at"],
                          "val_accuracy": m.get("val_accuracy"), "val_auc": m.get("val_auc"),
                          "labels_since": ls}

    liked = by_value.get(1.0, 0)
    collection = {
        "total": total, "liked": liked, "maybe": by_value.get(0.5, 0),
        "disliked": by_value.get(0.0, 0), "excluded": excluded,
        "unlabeled": total - labeled_any, "sources": n_sources,
        "anime": styles.get("anime", 0), "realistic": styles.get("realistic", 0),
    }
    labeled = total - collection["unlabeled"]
    # for the suggested action: most "stale" model (most new labels since train)
    labels_since = max((mm["labels_since"] for mm in models.values()), default=0)
    has_model = bool(models)

    # one suggested next action, by state
    if total == 0:
        nxt = {"action": "add", "label": "Add a folder of images to begin"}
    elif labeled < 50:
        nxt = {"action": "nav", "href": "/static/index.html",
               "label": f"Rate {50 - labeled} more images to unlock training"}
    elif not has_model:
        nxt = {"action": "train", "label": "Train your taste models"}
    elif labels_since >= 50:
        nxt = {"action": "train", "label": f"{labels_since} new ratings since last train — retrain"}
    elif liked >= 6:
        nxt = {"action": "nav", "href": "/static/tournament.html",
               "label": "Rank your favorites in a tournament"}
    else:
        nxt = {"action": "nav", "href": "/static/index.html", "label": "Keep rating"}
    return {"collection": collection, "models": models, "next": nxt}


@app.get("/api/styles")
def styles_counts():
    from . import styles
    return styles.counts()


@app.post("/api/styles/classify")
def styles_classify():
    from . import styles
    job = jobs.submit("classify", "style tagging",
                      lambda progress, cancel: styles.classify_styles(progress, cancel))
    return {"started": True, "job_id": job.id}


class SetStyleIn(BaseModel):
    image_ids: list[int] = Field(min_length=1, max_length=2000)
    style: str = Field(pattern="^(anime|realistic)$")


@app.post("/api/styles/set")
def styles_set(body: SetStyleIn):
    from . import styles

    with conn() as db:
        _require_images(db, body.image_ids)
    return {"updated": styles.set_style(body.image_ids, body.style)}


# ---- anime hand-quality scoring ----

@app.get("/api/hands/status")
def hands_status():
    """Hand model + scoring state for the Grid controls."""
    from . import hands

    head = hands.latest_head()
    with conn() as db:
        n_bad = db.execute("SELECT count(*) AS n FROM hand_labels WHERE label=0").fetchone()["n"]
        n_scored = db.execute("SELECT count(*) AS n FROM hand_scores").fetchone()["n"]
    return {"has_model": bool(head),
            "metrics": head.get("metrics") if head else None,
            "model": head.get("name") if head else None,
            "bad_labels": n_bad, "scored": n_scored}


@app.post("/api/hands/train")
def hands_train():
    from . import hands

    job = jobs.submit("hands", "training hand model",
                      lambda progress, cancel: hands.train(progress=progress, cancel=cancel))
    return {"started": True, "job_id": job.id}


@app.post("/api/hands/score")
def hands_score():
    from . import hands

    job = jobs.submit("hands", "scoring anime hands",
                      lambda progress, cancel: hands.score_all(progress, cancel))
    return {"started": True, "job_id": job.id}


class MarkHandsIn(BaseModel):
    image_ids: list[int] = Field(min_length=1, max_length=2000)
    bad: bool = True


@app.post("/api/hands/mark")
def hands_mark(body: MarkHandsIn):
    """Mark images as having bad hands (training negatives), or clear that mark.
    Good hands aren't marked — they come free from your likes."""
    with conn() as db:
        _require_images(db, body.image_ids)
        if body.bad:
            db.executemany(
                "INSERT INTO hand_labels (image_id, label, source) VALUES (?, 0, 'manual')"
                " ON CONFLICT(image_id) DO UPDATE SET label=0",
                [(i,) for i in body.image_ids])
        else:
            ph = ",".join("?" * len(body.image_ids))
            db.execute(f"DELETE FROM hand_labels WHERE image_id IN ({ph})", body.image_ids)
    return {"marked": len(body.image_ids), "bad": body.bad}


# Experimental hand-quality probe: serve detected hand crops + manifest for the
# labeling page (data/ is gitignored, so user images never enter the repo) and
# persist good/bad labels to a known file the scoring script reads.
_HAND_PROBE_DIR = DATA_DIR / "hand_probe"
_HAND_PROBE_DIR.mkdir(parents=True, exist_ok=True)


class HandLabelsIn(BaseModel):
    labels: dict[str, str] = Field(default_factory=dict)


@app.get("/api/hand_probe/labels")
def hand_probe_labels_get():
    f = _HAND_PROBE_DIR / "labels.json"
    return {"labels": json.loads(f.read_text()) if f.is_file() else {}}


@app.post("/api/hand_probe/labels")
def hand_probe_labels_set(body: HandLabelsIn):
    clean = {k: v for k, v in body.labels.items() if v in ("good", "bad")}
    (_HAND_PROBE_DIR / "labels.json").write_text(json.dumps(clean), encoding="utf-8")
    n_good = sum(1 for v in clean.values() if v == "good")
    return {"saved": len(clean), "good": n_good, "bad": len(clean) - n_good}


app.mount("/hand_probe", StaticFiles(directory=_HAND_PROBE_DIR), name="hand_probe")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
