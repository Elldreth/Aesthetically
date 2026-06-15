import numpy as np
from conftest import add_embedding, add_image, png_bytes

from app.db import get_conn


def _labeled_corpus(n=60, dim=16):
    """n images with embeddings clustered so liked/disliked are separable."""
    db = get_conn()
    rng = np.random.default_rng(0)
    like_dir = rng.standard_normal(dim).astype(np.float32)
    like_dir /= np.linalg.norm(like_dir)
    ids = []
    for i in range(n):
        image_id = add_image(db, str(i))
        liked = i % 2 == 0
        noise = rng.standard_normal(dim).astype(np.float32) * 0.4
        vec = (like_dir if liked else -like_dir) + noise
        vec /= np.linalg.norm(vec)
        from app.embed import MODEL_NAME

        db.execute(
            "INSERT OR REPLACE INTO embeddings (image_id, model, dim, vec) VALUES (?, ?, ?, ?)",
            (image_id, MODEL_NAME, dim, vec.astype(np.float32).tobytes()),
        )
        db.execute(
            "INSERT INTO labels (image_id, kind, value, source) VALUES (?, 'binary', ?, 'test')",
            (image_id, 1.0 if liked else 0.0),
        )
        ids.append(image_id)
    db.commit()
    db.close()
    return ids


def test_trained_at_uses_utc(tmp_db):
    """trained_at must track UTC (like SQLite datetime('now') on labels), or the
    'new ratings since last train' count compares across timezones and never
    resets after a retrain."""
    import json
    import time
    from datetime import datetime

    from app import taste

    _labeled_corpus()
    taste.train()
    head = json.loads((taste.MODELS_DIR / "taste_v1.json").read_text())
    fmt = "%Y-%m-%d %H:%M:%S"
    now_utc = time.strftime(fmt, time.gmtime())
    delta = abs((datetime.strptime(head["trained_at"], fmt)
                 - datetime.strptime(now_utc, fmt)).total_seconds())
    assert delta < 120, f"trained_at not UTC-aligned: {head['trained_at']} vs {now_utc}"


def test_taste_train_end_to_end(tmp_db):
    from app import taste

    _labeled_corpus()
    metrics = taste.train()
    assert metrics["model"] == "taste:v1"
    assert metrics["scored"] == 60
    assert metrics["val_n"] > 0
    assert (taste.MODELS_DIR / "taste_v1.json").is_file()
    # second train bumps version and fully replaces predictions
    metrics2 = taste.train()
    assert metrics2["model"] == "taste:v2"
    db = get_conn()
    models = {r[0] for r in db.execute("SELECT DISTINCT model FROM predictions")}
    assert models == {"taste:v2"}
    db.close()


def test_stale_prediction_cleanup(tmp_db, tmp_path):
    """store_embedding_and_score after a retrain must not leave two taste:% rows."""
    from app.scorer import store_embedding_and_score

    db = get_conn()
    image_id = add_image(db, "x")
    db.execute(
        "INSERT INTO predictions (image_id, model, score) VALUES (?, 'taste:v1', 0.4)",
        (image_id,),
    )
    db.commit()
    db.close()
    vec = np.ones(8, dtype=np.float32)
    store_embedding_and_score(image_id, vec, 0.9, "taste:v2")
    db = get_conn()
    rows = db.execute(
        "SELECT model, score FROM predictions WHERE image_id = ?", (image_id,)
    ).fetchall()
    assert len(rows) == 1 and rows[0]["model"] == "taste:v2"
    db.close()


class FakeArtifex:
    def __init__(self):
        self.train_calls = []

    def train(self, images, name, **kwargs):
        self.train_calls.append({"n_images": len(images), "name": name, **kwargs})
        return {"job_id": "fake-job", "state": "queued"}

    def train_status(self, job_id):
        return {"job_id": job_id, "state": "completed", "lora": "fake-lora"}


def test_train_taste_lora_with_fake_client(tmp_db, tmp_path):
    from app import studio

    db = get_conn()
    for i in range(15):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(png_bytes(color=(i * 10 % 255, 50, 90)))
        image_id = add_image(db, f"L{i}", path=str(p))
        add_embedding(db, image_id)
        db.execute(
            "INSERT INTO labels (image_id, kind, value, source) VALUES (?, 'binary', 1.0, 'test')",
            (image_id,),
        )
    db.commit()
    db.close()

    fake = FakeArtifex()
    out = studio.train_taste_lora("test-lora", max_images=12, preset="subtle", client=fake)
    assert out["dataset_size"] == 12
    assert fake.train_calls[0]["n_images"] == 12
    # steps derive from image count: 12 * 40/img = 480, clamped up to the 600 floor
    assert out["steps"] == 600
    assert fake.train_calls[0]["steps"] == 600
    db = get_conn()
    run = db.execute("SELECT * FROM training_runs WHERE id = ?", (out["run_id"],)).fetchone()
    assert run["status"] == "running"
    import json

    fp = json.loads(run["dataset_fingerprint_json"])
    assert fp["n"] == 12 and len(fp["image_ids"]) == 12
    db.close()

    status = studio.poll_run(out["run_id"], client=fake)
    assert status["job"]["state"] == "completed"
    db = get_conn()
    run = db.execute("SELECT * FROM training_runs WHERE id = ?", (out["run_id"],)).fetchone()
    assert run["status"] == "done" and run["artifact_path"] == "fake-lora"
    db.close()


def test_train_taste_lora_filters_by_style_and_picks_checkpoint(tmp_db, tmp_path):
    import json

    from app import studio

    db = get_conn()
    anime_ids = []
    for i in range(12):
        for style, suffix in (("anime", "A"), ("realistic", "R")):
            p = tmp_path / f"{style}{i}.png"
            p.write_bytes(png_bytes(color=((i * 9) % 255, 40, 90)))
            image_id = add_image(db, f"{suffix}{i}", path=str(p))
            add_embedding(db, image_id)
            db.execute("INSERT INTO labels (image_id, kind, value, source)"
                       " VALUES (?, 'binary', 1.0, 'test')", (image_id,))
            db.execute("INSERT INTO image_styles (image_id, style, source)"
                       " VALUES (?, ?, 'manual')", (image_id, style))
            if style == "anime":
                anime_ids.append(image_id)
    db.commit()
    db.close()

    # liked_counts reflects the per-style split
    assert studio.liked_counts()["anime"] == 12
    assert studio.liked_counts()["realistic"] == 12

    fake = FakeArtifex()
    out = studio.train_taste_lora("anime-lora", max_images=20, style="anime", client=fake)
    # base model defaulted to the anime checkpoint...
    assert out["model"] == studio.STYLE_CHECKPOINTS["anime"]
    assert fake.train_calls[0]["model"] == studio.STYLE_CHECKPOINTS["anime"]
    # ...and the dataset is anime-only (none of the 12 realistic images leaked in)
    db = get_conn()
    fp = json.loads(db.execute(
        "SELECT dataset_fingerprint_json FROM training_runs WHERE id=?",
        (out["run_id"],)).fetchone()["dataset_fingerprint_json"])
    db.close()
    assert len(fp["image_ids"]) == 12
    assert set(fp["image_ids"]) <= set(anime_ids)

    # an explicit model overrides the style default
    out2 = studio.train_taste_lora("anime-lora-2", max_images=20, style="anime",
                                   model="juggernaut-xl-ragnarok", client=fake)
    assert out2["model"] == "juggernaut-xl-ragnarok"


def test_train_taste_lora_requires_enough_likes(tmp_db):
    from app import studio

    try:
        studio.train_taste_lora("x", max_images=10, client=FakeArtifex())
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "rate more" in str(e)
