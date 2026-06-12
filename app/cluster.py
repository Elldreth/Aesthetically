"""Cluster liked images by visual style in SigLIP embedding space.

A single "taste LoRA" from heterogeneous likes has no coherent style to bind
to. Clustering the likes first finds the groups that DO share a look (a curvy
photoreal cluster, an anime cluster, a landscape cluster, …); each coherent
cluster is a good style-LoRA dataset. HDBSCAN labels sparse, incoherent likes
as noise so we don't train on them.
"""
from __future__ import annotations

import numpy as np

from .db import conn
from .embed import MODEL_NAME, load_vectors


def cluster_likes(min_cluster_size: int = 15) -> list[dict]:
    """Coherent style clusters among liked (non-near-dup) images, most-cohesive
    first. Each cluster: {cluster, size, cohesion, image_ids, samples}."""
    from sklearn.cluster import HDBSCAN

    with conn() as db:
        liked = [
            r["image_id"] for r in db.execute(
                "SELECT image_id FROM current_labels c WHERE c.kind='binary' AND c.value=1.0"
                " AND NOT EXISTS (SELECT 1 FROM near_dups n WHERE n.image_id = c.image_id)"
            )
        ]
        ids, mat = load_vectors(db, MODEL_NAME, image_ids=liked)
    if len(ids) < min_cluster_size:
        return []

    # SigLIP embeddings are ~1152-dim; HDBSCAN suffers from distance
    # concentration that high, so PCA-reduce first (standard practice). The
    # reduced space is only for finding groups — cohesion is measured back in
    # the full embedding space below.
    from sklearn.decomposition import PCA

    n_comp = min(50, mat.shape[0] - 1, mat.shape[1])
    reduced = PCA(n_components=n_comp).fit_transform(mat) if n_comp >= 2 else mat
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)
    index = {int(v): k for k, v in enumerate(ids)}
    groups: dict[int, list[int]] = {}
    for image_id, lab in zip(ids, labels):
        if lab < 0:
            continue  # noise — not a coherent style
        groups.setdefault(int(lab), []).append(int(image_id))

    out = []
    for lab, members in groups.items():
        rows = np.array([index[m] for m in members])
        sub = mat[rows]
        centroid = sub.mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        sims = sub @ centroid
        cohesion = float(sims.mean())
        order = np.argsort(-sims)
        samples = [members[i] for i in order[:6]]          # closest to centroid
        out.append({
            "cluster": lab, "size": len(members),
            "cohesion": round(cohesion, 3),
            "image_ids": members, "samples": samples,
        })
    out.sort(key=lambda c: -c["cohesion"])
    return out
