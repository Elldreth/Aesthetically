import base64

from conftest import add_image, png_bytes

from app.db import get_conn


def _setup_images(n=4):
    db = get_conn()
    ids = [add_image(db, str(i)) for i in range(n)]
    db.commit()
    db.close()
    return ids


def _session(client) -> int:
    return client.post("/api/sessions", json={"name": "t"}).json()["session_id"]


def test_token_required_for_mutations(client):
    ids = _setup_images(1)
    no_token = {k: v for k, v in client.headers.items() if k.lower() != "x-aesth-token"}
    r = client.post("/api/label", json={"image_id": ids[0], "value": 1},
                    headers={"X-Aesth-Token": "wrong"})
    assert r.status_code == 403
    # GETs stay open
    assert client.get("/api/stats").status_code == 200


def test_unknown_host_rejected(client):
    r = client.get("/api/stats", headers={"host": "evil.example.com"})
    assert r.status_code == 421


def test_label_current_semantics_and_undo(client):
    ids = _setup_images(1)
    sid = _session(client)
    client.post("/api/label", json={"image_id": ids[0], "value": 1, "session_id": sid})
    client.post("/api/label", json={"image_id": ids[0], "value": 0, "session_id": sid})
    stats = client.get("/api/stats").json()
    assert stats["disliked"] == 1 and stats["liked"] == 0  # latest wins
    r = client.post("/api/undo", json={"session_id": sid})
    assert r.json()["value"] == 0.0
    stats = client.get("/api/stats").json()
    assert stats["liked"] == 1 and stats["disliked"] == 0  # reverted to prior


def test_undo_requires_and_respects_session(client):
    ids = _setup_images(2)
    s1, s2 = _session(client), _session(client)
    client.post("/api/label", json={"image_id": ids[0], "value": 1, "session_id": s1})
    client.post("/api/label", json={"image_id": ids[1], "value": 0, "session_id": s2})
    assert client.post("/api/undo", json={}).status_code == 422
    r = client.post("/api/undo", json={"session_id": s1})
    assert r.json()["image_id"] == ids[0]  # s2's later label untouched
    stats = client.get("/api/stats").json()
    assert stats["disliked"] == 1 and stats["liked"] == 0


def test_queue_excludes_labeled_excluded_neardup(client):
    ids = _setup_images(4)
    sid = _session(client)
    client.post("/api/label", json={"image_id": ids[0], "value": 1, "session_id": sid})
    client.post("/api/exclude", json={"image_id": ids[1], "session_id": sid})
    db = get_conn()
    db.execute("INSERT INTO near_dups (image_id, canonical_id, method) VALUES (?, ?, 'phash')",
               (ids[2], ids[3]))
    db.commit()
    db.close()
    queue = client.get("/api/queue").json()["items"]
    assert [i["id"] for i in queue] == [ids[3]]
    assert client.get("/api/queue?mode=bogus").status_code == 422


def test_stats_not_double_subtracted(client):
    ids = _setup_images(2)
    sid = _session(client)
    client.post("/api/label", json={"image_id": ids[0], "value": 0, "session_id": sid})
    client.post("/api/exclude", json={"image_id": ids[0], "session_id": sid})
    stats = client.get("/api/stats").json()
    assert stats["unlabeled"] == 1  # ids[1]; ids[0] counted once despite two labels


def test_bulk_label_validates_ids(client):
    ids = _setup_images(1)
    sid = _session(client)
    r = client.post("/api/labels/bulk",
                    json={"image_ids": [ids[0], 99999], "value": 0, "session_id": sid})
    assert r.status_code == 404
    stats = client.get("/api/stats").json()
    assert stats["disliked"] == 0  # nothing partially applied


def test_ingest_endpoint_dedup_and_label(client):
    data = base64.b64encode(png_bytes(color=(9, 9, 9))).decode()
    r1 = client.post("/api/ingest", json={"data_b64": data, "value": 1,
                                          "image_url": "https://a.example/x.png"}).json()
    r2 = client.post("/api/ingest", json={"data_b64": data,
                                          "image_url": "https://b.example/y.png"}).json()
    assert r1["created"] is True and r2["created"] is False
    assert r1["image_id"] == r2["image_id"]
    bad = client.post("/api/ingest", json={"data_b64": base64.b64encode(b"junk").decode()})
    assert bad.status_code == 422
