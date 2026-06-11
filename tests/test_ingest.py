import threading

from conftest import png_bytes

from app.db import get_conn
from app.ingest import parse_sd_parameters, register_bytes


def test_register_same_bytes_twice(tmp_db, tmp_path):
    data = png_bytes()
    store = tmp_path / "store"
    db = get_conn()
    id1, created1 = register_bytes(db, data, store_dir=store)
    id2, created2 = register_bytes(db, data, store_dir=store)
    db.commit()
    assert created1 is True and created2 is False
    assert id1 == id2
    assert len(list(store.iterdir())) == 1
    db.close()


def test_concurrent_ingest_same_new_bytes(tmp_db, tmp_path):
    """Two clients race to ingest the same new image: both must succeed and
    exactly one row may exist (regression: IntegrityError was masked as 422)."""
    data = png_bytes(color=(1, 2, 3))
    store = tmp_path / "store"
    results, errors = [], []
    barrier = threading.Barrier(2)

    def worker():
        db = get_conn()
        try:
            barrier.wait()
            image_id, _created = register_bytes(db, data, store_dir=store)
            db.commit()
            results.append(image_id)
        except Exception as e:  # noqa: BLE001 - test records any failure
            errors.append(e)
        finally:
            db.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent ingest raised: {errors}"
    assert len(set(results)) == 1
    db = get_conn()
    n = db.execute("SELECT count(*) FROM images").fetchone()[0]
    assert n == 1
    db.close()


def test_parse_sd_parameters_full():
    raw = ("a castle on a hill\nNegative prompt: blurry, ugly\n"
           "Steps: 30, Sampler: DPM++ 2M, Model hash: abc123de, Seed: 42")
    out = parse_sd_parameters(raw)
    assert out["prompt"] == "a castle on a hill"
    assert out["negative_prompt"] == "blurry, ugly"
    assert out["model_hash"] == "abc123de"
    assert out["seed"] == "42"


def test_parse_sd_parameters_no_negative():
    out = parse_sd_parameters("just a prompt\nSteps: 20, Seed: 7")
    assert out["prompt"] == "just a prompt"
    assert out["negative_prompt"] is None
    assert out["seed"] == "7"


def test_parse_sd_parameters_empty():
    out = parse_sd_parameters("")
    assert out == {"prompt": None, "negative_prompt": None,
                   "model_hash": None, "seed": None}
