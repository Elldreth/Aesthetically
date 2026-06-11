"""Image ingestion: hashing, metadata extraction, upsert into the DB.

Files are never moved or modified; ingestion only reads bytes and records rows.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from io import BytesIO
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# A1111/SD-webui "parameters" text chunk, best effort:
#   <prompt>
#   Negative prompt: <negative>
#   Steps: 30, Sampler: ..., Model hash: abcd1234, Seed: 12345, ...
_RE_MODEL_HASH = re.compile(r"Model hash:\s*([0-9a-fA-F]+)")
_RE_SEED = re.compile(r"Seed:\s*(\d+)")


def parse_sd_parameters(raw: str) -> dict:
    out = {"prompt": None, "negative_prompt": None, "model_hash": None, "seed": None}
    if not raw:
        return out
    neg_idx = raw.find("Negative prompt:")
    steps_idx = raw.find("\nSteps:")
    if neg_idx >= 0:
        out["prompt"] = raw[:neg_idx].strip() or None
        neg_end = steps_idx if steps_idx > neg_idx else len(raw)
        out["negative_prompt"] = raw[neg_idx + len("Negative prompt:"):neg_end].strip() or None
    elif steps_idx >= 0:
        out["prompt"] = raw[:steps_idx].strip() or None
    else:
        out["prompt"] = raw.strip() or None
    m = _RE_MODEL_HASH.search(raw)
    if m:
        out["model_hash"] = m.group(1)
    m = _RE_SEED.search(raw)
    if m:
        out["seed"] = m.group(1)
    return out


def inspect_image(data: bytes) -> dict:
    """Dimensions, format, and any SD generation metadata from image bytes."""
    info: dict = {"width": None, "height": None, "format": None, "gen_params_raw": None}
    with Image.open(BytesIO(data)) as img:
        info["width"], info["height"] = img.size
        info["format"] = img.format
        raw = img.info.get("parameters")
        if raw:
            info["gen_params_raw"] = raw
            info.update(parse_sd_parameters(raw))
    return info


def upsert_image(conn: sqlite3.Connection, path: Path) -> int | None:
    """Register a local file. Returns image id, or None if unreadable.

    Existing rows are matched by sha256; a new source row is added either way.
    Metadata fields are only filled in when currently NULL (first writer wins).
    """
    try:
        data = path.read_bytes()
        info = inspect_image(data)
    except Exception:
        return None
    sha = hashlib.sha256(data).hexdigest()

    row = conn.execute("SELECT id FROM images WHERE sha256 = ?", (sha,)).fetchone()
    if row:
        image_id = row["id"]
        if info.get("gen_params_raw"):
            conn.execute(
                """UPDATE images SET
                     prompt = COALESCE(prompt, ?),
                     negative_prompt = COALESCE(negative_prompt, ?),
                     model_hash = COALESCE(model_hash, ?),
                     seed = COALESCE(seed, ?),
                     gen_params_raw = COALESCE(gen_params_raw, ?)
                   WHERE id = ?""",
                (info.get("prompt"), info.get("negative_prompt"), info.get("model_hash"),
                 info.get("seed"), info.get("gen_params_raw"), image_id),
            )
    else:
        cur = conn.execute(
            """INSERT INTO images (sha256, width, height, format, file_size,
                                   prompt, negative_prompt, model_hash, seed, gen_params_raw)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sha, info["width"], info["height"], info["format"], len(data),
             info.get("prompt"), info.get("negative_prompt"), info.get("model_hash"),
             info.get("seed"), info.get("gen_params_raw")),
        )
        image_id = cur.lastrowid

    conn.execute(
        """INSERT INTO image_sources (image_id, kind, location)
           VALUES (?, 'local', ?)
           ON CONFLICT (image_id, location) DO UPDATE SET last_verified = datetime('now')""",
        (image_id, str(path.resolve())),
    )
    return image_id


def iter_image_files(folder: Path, recursive: bool = True) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
