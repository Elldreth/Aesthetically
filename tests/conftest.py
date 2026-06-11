import io
import os

# must be set before app.main is imported (read at module import)
os.environ.setdefault("AESTH_ALLOWED_HOSTS",
                      "testserver,127.0.0.1:8787,localhost:8787")

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Point the whole app at a fresh database + data dir."""
    import app.db as db_mod

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DATA_DIR", tmp_path)
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main_mod, "TOKEN_PATH", tmp_path / "token.txt")
    import app.taste as taste_mod

    monkeypatch.setattr(taste_mod, "MODELS_DIR", tmp_path / "models")
    return db_path


@pytest.fixture()
def client(tmp_db):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        token = app.state.token
        c.headers["X-Aesth-Token"] = token
        yield c


def png_bytes(color=(120, 30, 200), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def add_image(db, sha_suffix: str, path: str | None = None) -> int:
    cur = db.execute(
        "INSERT INTO images (sha256, width, height, format) VALUES (?, 64, 64, 'PNG')",
        (f"{'0' * 56}{sha_suffix:>08}",),
    )
    image_id = cur.lastrowid
    if path:
        db.execute(
            "INSERT INTO image_sources (image_id, kind, location) VALUES (?, 'local', ?)",
            (image_id, path),
        )
    return image_id


def add_embedding(db, image_id: int, dim: int = 16, seed: int | None = None):
    from app.embed import MODEL_NAME

    rng = np.random.default_rng(seed if seed is not None else image_id)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    db.execute(
        "INSERT OR REPLACE INTO embeddings (image_id, model, dim, vec) VALUES (?, ?, ?, ?)",
        (image_id, MODEL_NAME, dim, vec.tobytes()),
    )
    return vec
