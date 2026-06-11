import numpy as np
from conftest import png_bytes

from app.db import get_conn


def _fake_head():
    return {"name": "taste:test", "coef": [0.1] * 16, "intercept": 0.0}


def _new_scan(folder) -> int:
    import app.scan as scan_mod

    db = get_conn()
    sid = db.execute(
        "INSERT INTO scans (path, taste_model, embed_model) VALUES (?, ?, ?)",
        (str(folder), "taste:test", scan_mod.MODEL_NAME),
    ).lastrowid
    db.commit()
    db.close()
    return sid


def test_scan_caches_embeddings_and_persists(tmp_db, tmp_path, monkeypatch):
    import app.scan as scan_mod

    calls = {"n": 0}

    def fake_embed(pils):
        calls["n"] += len(pils)
        v = np.tile(np.linspace(0, 1, 16).astype(np.float32), (len(pils), 1))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(scan_mod, "embed_pil", fake_embed)
    monkeypatch.setattr(scan_mod, "latest_head", _fake_head)

    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in range(5):
        (folder / f"{i}.png").write_bytes(png_bytes(color=(i * 40, i * 30, i * 20)))

    sid = _new_scan(folder)
    results = scan_mod.run_scan(folder, sid, {})
    assert len(results) == 5
    assert calls["n"] == 5  # all five embedded on first pass (cache misses)

    db = get_conn()
    assert db.execute("SELECT count(*) FROM scan_items WHERE scan_id = ?",
                      (sid,)).fetchone()[0] == 5
    assert db.execute("SELECT count(*) FROM scan_cache").fetchone()[0] == 5
    assert db.execute("SELECT count FROM scans WHERE id = ?", (sid,)).fetchone()[0] == 5
    db.close()

    # a second scan of the same folder must hit the cache — no re-embedding
    calls["n"] = 0
    sid2 = _new_scan(folder)
    scan_mod.run_scan(folder, sid2, {})
    assert calls["n"] == 0
    db = get_conn()
    assert db.execute("SELECT count(*) FROM scan_cache").fetchone()[0] == 5  # unchanged
    db.close()


def test_scan_results_sorted_desc(tmp_db, tmp_path, monkeypatch):
    import app.scan as scan_mod

    # distinct scores via per-image embeddings tied to the red channel
    def fake_embed(pils):
        out = []
        for p in pils:
            r = p.getpixel((0, 0))[0] / 255.0
            v = np.full(16, r, dtype=np.float32)
            out.append(v / (np.linalg.norm(v) or 1))
        return np.array(out)

    monkeypatch.setattr(scan_mod, "embed_pil", fake_embed)
    monkeypatch.setattr(scan_mod, "latest_head", _fake_head)

    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in range(4):
        (folder / f"{i}.png").write_bytes(png_bytes(color=(i * 60, 0, 0)))

    sid = _new_scan(folder)
    results = scan_mod.run_scan(folder, sid, {})
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
