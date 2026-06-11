import numpy as np
from conftest import add_image

from app.db import get_conn
from app.embed import MODEL_NAME


def _liked_with_vec(db, image_id: int, vec: np.ndarray):
    db.execute(
        "INSERT INTO labels (image_id, kind, value, source) VALUES (?, 'binary', 1.0, 'test')",
        (image_id,),
    )
    db.execute(
        "INSERT OR REPLACE INTO embeddings (image_id, model, dim, vec) VALUES (?, ?, ?, ?)",
        (image_id, MODEL_NAME, vec.shape[0], vec.astype(np.float32).tobytes()),
    )


def test_cluster_finds_two_style_groups(tmp_db):
    from app.cluster import cluster_likes

    db = get_conn()
    rng = np.random.default_rng(0)
    dirs = [np.eye(16, dtype=np.float32)[0], np.eye(16, dtype=np.float32)[8]]
    members = {0: [], 1: []}
    n = 1
    for g, d in enumerate(dirs):
        for _ in range(20):
            v = d + rng.standard_normal(16).astype(np.float32) * 0.05
            v /= np.linalg.norm(v)
            iid = add_image(db, str(n)); n += 1
            _liked_with_vec(db, iid, v)
            members[g].append(iid)
    db.commit()
    db.close()

    clusters = cluster_likes(min_cluster_size=10)
    assert len(clusters) == 2
    # every cluster is internally consistent: its members all come from one group
    group_of = {iid: g for g, ids in members.items() for iid in ids}
    for c in clusters:
        groups = {group_of[i] for i in c["image_ids"]}
        assert len(groups) == 1
        assert c["size"] >= 10
        assert 0.0 <= c["cohesion"] <= 1.0
        assert len(c["samples"]) <= 6


def test_cluster_empty_when_too_few_likes(tmp_db):
    from app.cluster import cluster_likes

    assert cluster_likes(min_cluster_size=15) == []
