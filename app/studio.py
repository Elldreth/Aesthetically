"""The Artifex closed loop: generate → score → curate → train → evaluate.

Aesthetically is the taste oracle; Artifex (http://127.0.0.1:7860) is the
generator/trainer. Everything here is plain HTTP against its documented API.
"""
from __future__ import annotations

import base64
import json
import os
import random

import numpy as np
from PIL import Image
from io import BytesIO

from .artifex_client import ArtifexClient
from .db import DATA_DIR, conn
from .embed import MODEL_NAME, load_vectors, text_features
from .scorer import latest_head, score_images, store_embedding_and_score

GEN_DIR = DATA_DIR / "generated"

# Default taste-LoRA strength. 0.6 beat 0.8 in side-by-side tests on a matched
# checkpoint — it refines toward the trained style without overpowering the
# base (0.8 fought composition and only broke even).
DEFAULT_LORA_WEIGHT = 0.6

# LoRA training presets — values from a 2026 research pass over kohya/Civitai/HF
# diffusers guidance (see commit message). Total steps are DERIVED:
# steps = clamp(images * steps_per_image, STEPS_MIN, STEPS_MAX). STYLE LoRAs
# overfit faster than character LoRAs, so steps/image runs well below the usual
# ~100. alpha = rank/2 (the dominant style convention) and LRs are tuned for it.
LORA_PRESETS = {
    "subtle":   {"steps_per_image": 40,  "lr": 8e-5,  "rank": 16,
                 "use": "light flavor; keeps the base model dominant, low overfit risk"},
    "balanced": {"steps_per_image": 75,  "lr": 1e-4,  "rank": 32,
                 "use": "recommended default for most aesthetic styles"},
    "strong":   {"steps_per_image": 110, "lr": 1.2e-4, "rank": 64,
                 "use": "dense/complex styles needing more capacity (realism > anime)"},
}
DEFAULT_PRESET = "balanced"
DEFAULT_MAX_IMAGES = 60
STEPS_MIN, STEPS_MAX = 600, 3000   # <600 underfits; >3000 SDXL style LoRAs overfit

# Which base checkpoint a style's LoRA trains against. A style LoRA only helps
# on a matched checkpoint (validated: anime LoRA improved hoseki, not juggernaut),
# so the style choice also picks a sensible default base model — overridable.
STYLE_CHECKPOINTS = {
    "anime": "hoseki-noobai-anime",
    "realistic": "juggernaut-xl-ragnarok",
}


def _resolve_training(preset: str, n_images: int, steps: int | None,
                      lr: float | None, rank: int | None) -> dict:
    """Turn a preset + actual image count into concrete (steps, lr, rank, alpha),
    with explicit overrides winning. Steps derive from the image count."""
    p = LORA_PRESETS.get(preset or DEFAULT_PRESET, LORA_PRESETS[DEFAULT_PRESET])
    resolved_steps = steps if steps else max(STEPS_MIN, min(STEPS_MAX,
                                             n_images * p["steps_per_image"]))
    resolved_rank = rank or p["rank"]
    return {"preset": preset or DEFAULT_PRESET, "steps": resolved_steps,
            "lr": lr if lr is not None else p["lr"],
            "rank": resolved_rank, "alpha": max(1, resolved_rank // 2),
            "steps_per_image": p["steps_per_image"]}

_shared_client: ArtifexClient | None = None


def _get_client(client: ArtifexClient | None) -> ArtifexClient:
    """One long-lived process client (httpx.Client is thread-safe) — callers
    that pass their own client keep ownership of it."""
    global _shared_client
    if client is not None:
        return client
    if _shared_client is None:
        _shared_client = ArtifexClient()
    return _shared_client


def _assert_artifex_free() -> None:
    """Artifex shares one GPU — generation blocks while a LoRA trains. Fail
    fast with a clear message instead of a multi-minute timeout."""
    with conn() as db:
        row = db.execute(
            "SELECT id, name FROM training_runs WHERE status = 'running'"
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row:
        raise RuntimeError(
            f"Artifex is training run #{row['id']} ({row['name']}) — "
            "wait for it to finish before generating/evaluating")


def best_of_n(prompt: str, n: int = 4, model: str | None = None,
              size: str = "832x1216", loras: list | None = None,
              client: ArtifexClient | None = None) -> list[dict]:
    """Generate n seeds of a prompt, ingest each with authoritative metadata,
    score with the taste head, return ranked best-first."""
    _assert_artifex_free()
    client = _get_client(client)
    head = latest_head()
    results = []
    for _ in range(n):
        seed = random.randrange(2**31)
        payload = {"prompt": f"{prompt} --seed {seed}", "size": size}
        if model:
            payload["model"] = model
        if loras:
            payload["loras"] = loras
        out = client.generate(**payload)
        data = base64.b64decode(out["data"][0]["b64_json"])
        results.append({"seed": seed, "data": data})

    ranked = []
    with conn() as db:
        from .ingest import register_bytes

        for r in results:
            image_id, created = register_bytes(
                db, r["data"], store_dir=GEN_DIR,
                gen_meta={"prompt": prompt, "model_hash": model or "artifex-default",
                          "seed": str(r["seed"])},
            )
            ranked.append({"image_id": image_id, "seed": r["seed"]})
        db.commit()
    pils = [Image.open(BytesIO(r["data"])) for r in results]
    if head:
        scores, vecs = score_images(pils, head, return_vecs=True)
        for item, s, v in zip(ranked, scores, vecs):
            item["score"] = round(float(s), 4)
            store_embedding_and_score(item["image_id"], v, s, head["name"])
        ranked.sort(key=lambda x: -x["score"])
    return ranked


def _diversity_select(ids: np.ndarray, mat: np.ndarray, k: int,
                      seed_order: list[int]) -> list[int]:
    """Greedy max-min selection: start from the top-ranked image, then always
    add the candidate farthest from everything already chosen."""
    if len(ids) <= k:
        return [int(i) for i in ids]
    index = {int(v): j for j, v in enumerate(ids)}
    chosen = [index[seed_order[0]]]
    rest = [index[i] for i in seed_order[1:] if int(ids[index[i]]) != int(ids[chosen[0]])]
    while len(chosen) < k and rest:
        sims = mat[rest] @ mat[chosen].T          # (rest, chosen)
        farthest = int(np.argmin(sims.max(axis=1)))
        chosen.append(rest.pop(farthest))
    return [int(ids[j]) for j in chosen]


def liked_counts() -> dict:
    """Eligible (liked, non-near-dup) image counts per style — what a taste LoRA
    can actually draw from. Powers the training readout."""
    with conn() as db:
        base = ("FROM images i"
                " JOIN current_labels c ON c.image_id = i.id"
                "  AND c.kind = 'binary' AND c.value = 1.0"
                " WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)")
        by = {r["style"]: r["n"] for r in db.execute(
            "SELECT ist.style AS style, count(*) AS n"
            " FROM images i"
            " JOIN current_labels c ON c.image_id = i.id"
            "  AND c.kind = 'binary' AND c.value = 1.0"
            " JOIN image_styles ist ON ist.image_id = i.id"
            " WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)"
            " GROUP BY ist.style").fetchall()}
        total = db.execute(f"SELECT count(*) AS n {base}").fetchone()["n"]
        # liked-anime pools under each hand filter, for the training readout
        from .select import hand_clause
        anime_base = (
            "FROM images i"
            " JOIN current_labels c ON c.image_id = i.id AND c.kind='binary' AND c.value=1.0"
            " JOIN image_styles ist ON ist.image_id = i.id AND ist.style='anime'"
            " WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)")
        good = db.execute(f"SELECT count(*) AS n {anime_base}{hand_clause('good')}").fetchone()["n"]
        not_bad = db.execute(f"SELECT count(*) AS n {anime_base}{hand_clause('not_bad')}").fetchone()["n"]
    return {"anime": by.get("anime", 0), "realistic": by.get("realistic", 0), "all": total,
            "anime_good_hands": good, "anime_not_bad_hands": not_bad}


def build_taste_dataset(k: int = 40, style: str | None = None,
                        hand_filter: str | None = None) -> list[dict]:
    """Pick the k best-loved, maximally diverse images for a style LoRA.

    With ``style`` ('anime'/'realistic'), the dataset is restricted to images
    tagged that style. ``hand_filter`` ('good' = only good-hands-tagged,
    'not_bad' = exclude bad-hands-tagged) keeps bad hands out of the LoRA.
    Ranking: Bradley-Terry strength when pairwise votes exist, else taste score,
    else recency. Diversity: greedy max-min in SigLIP space."""
    from .select import hand_clause

    style_join = ("JOIN image_styles ist ON ist.image_id = i.id AND ist.style = ?"
                  if style in ("anime", "realistic") else "")
    params = (style,) if style_join else ()
    with conn() as db:
        rows = db.execute(
            f"""SELECT i.id,
                      (SELECT score FROM predictions p WHERE p.image_id = i.id
                       AND p.model LIKE 'taste:%') AS taste,
                      (SELECT count(*) FROM labels w WHERE w.kind = 'pairwise'
                       AND w.image_id = i.id) AS wins,
                      (SELECT count(*) FROM labels l WHERE l.kind = 'pairwise'
                       AND l.opponent_image_id = i.id) AS losses,
                      (SELECT location FROM image_sources s WHERE s.image_id = i.id
                       AND s.kind = 'local' LIMIT 1) AS path
               FROM images i
               JOIN current_labels c ON c.image_id = i.id AND c.kind = 'binary' AND c.value = 1.0
               {style_join}
               WHERE NOT EXISTS (SELECT 1 FROM near_dups d WHERE d.image_id = i.id)
               {hand_clause(hand_filter)}""",
            params,
        ).fetchall()
        cands = [dict(r) for r in rows if r["path"]]
        for c in cands:
            n = c["wins"] + c["losses"]
            c["bt"] = (c["wins"] + 1) / (n + 2)            # smoothed win rate
            c["rank_key"] = (c["bt"] if n else 0.5, c["taste"] or 0.5)
        cands.sort(key=lambda c: c["rank_key"], reverse=True)
        pool = cands[: max(k * 3, 60)]                      # quality floor first
        ids, mat = load_vectors(db, MODEL_NAME, image_ids=[c["id"] for c in pool])
        order = [c["id"] for c in pool if c["id"] in set(int(i) for i in ids)]
        keep = set(_diversity_select(ids, mat, k, order))
    return [c for c in cands if c["id"] in keep]


def _image_paths(db, image_ids: list[int]) -> list[tuple[int, str]]:
    out = []
    for image_id in image_ids:
        row = db.execute(
            "SELECT location FROM image_sources WHERE image_id = ? AND kind = 'local' LIMIT 1",
            (image_id,),
        ).fetchone()
        if row and os.path.isfile(row["location"]):
            out.append((image_id, row["location"]))
    return out


def submit_lora(name: str, image_ids: list[int], *, preset: str = DEFAULT_PRESET,
                max_images: int = DEFAULT_MAX_IMAGES, model: str | None = None,
                steps: int | None = None, lr: float | None = None,
                rank: int | None = None, client: ArtifexClient | None = None) -> dict:
    """Submit a style-LoRA job to Artifex from an explicit image-id list.
    Caps to max_images by greedy diversity selection, then DERIVES total steps
    from the actual image count via the chosen preset (overridable). Returns the
    real settings used so the UI never has to guess."""
    from pathlib import Path

    client = _get_client(client)
    with conn() as db:
        pairs = _image_paths(db, image_ids)
        if len(pairs) < 10:
            raise RuntimeError(f"only {len(pairs)} usable images — need at least 10")
        ids = [i for i, _ in pairs]
        path_by_id = dict(pairs)
        kept_ids, mat = load_vectors(db, MODEL_NAME, image_ids=ids)
        if len(kept_ids) > max_images:
            order = [int(i) for i in kept_ids]
            keep = set(_diversity_select(kept_ids, mat, max_images, order))
            ids = [i for i in ids if i in keep]
            _, mat = load_vectors(db, MODEL_NAME, image_ids=ids)

    t = _resolve_training(preset, len(ids), steps, lr, rank)

    images = []
    for image_id in ids:
        raw = Path(path_by_id[image_id]).read_bytes()
        images.append("data:image/png;base64," + base64.b64encode(raw).decode())

    config = {
        "name": name, "steps": t["steps"], "rank": t["rank"], "alpha": t["alpha"],
        "lr": t["lr"], "auto_caption": True, "prune_tags": True, "sampling": "balance",
        # style recipe: content-diverse, captions describe content so style binds
    }
    if model:
        config["model"] = model            # base checkpoint to train against
    job = client.train(images=images, captions=[""] * len(images), **config)

    with conn() as db:
        sims = mat @ mat.T
        fingerprint = {
            "n": len(ids), "image_ids": ids, "preset": t["preset"],
            "steps": t["steps"], "lr": t["lr"], "rank": t["rank"],
            "mean_pairwise_sim": round(float(sims[np.triu_indices(len(ids), 1)].mean()), 4)
            if len(ids) > 1 else None,
        }
        cur = db.execute(
            """INSERT INTO training_runs (name, config_json, dataset_fingerprint_json,
                                          status, started_at)
               VALUES (?, ?, ?, 'running', datetime('now'))""",
            (name, json.dumps({**config, "job_id": job.get("job_id")}),
             json.dumps(fingerprint)),
        )
        run_id = cur.lastrowid
    return {"run_id": run_id, "job_id": job.get("job_id"), "dataset_size": len(ids),
            "steps": t["steps"], "lr": t["lr"], "rank": t["rank"], "preset": t["preset"],
            "model": model, "state": job.get("state")}


def train_taste_lora(name: str, max_images: int = DEFAULT_MAX_IMAGES,
                     preset: str = DEFAULT_PRESET, model: str | None = None,
                     style: str | None = None, hand_filter: str | None = None,
                     steps: int | None = None, lr: float | None = None,
                     rank: int | None = None,
                     client: ArtifexClient | None = None) -> dict:
    """Submit a taste-LoRA job from the curated dataset for a style.

    ``style`` restricts the dataset to that style's likes and, when ``model`` is
    not given, selects the matching base checkpoint (anime → hoseki, etc.).
    ``hand_filter`` keeps bad-hands images out (see build_taste_dataset)."""
    dataset = build_taste_dataset(max_images, style=style, hand_filter=hand_filter)
    if len(dataset) < 10:
        scope = f" {style}" if style in ("anime", "realistic") else ""
        raise RuntimeError(
            f"only {len(dataset)} usable{scope} liked images — rate more first")
    if model is None and style in STYLE_CHECKPOINTS:
        model = STYLE_CHECKPOINTS[style]
    return submit_lora(name, [c["id"] for c in dataset], preset=preset,
                       max_images=max_images, model=model, steps=steps, lr=lr,
                       rank=rank, client=client)


def delete_lora(name: str, client: ArtifexClient | None = None) -> dict:
    """Permanently delete a trained LoRA from Artifex and drop its run records."""
    client = _get_client(client)
    artifex = client.delete_lora(name)
    with conn() as db:
        cur = db.execute("DELETE FROM training_runs WHERE artifact_path = ?", (name,))
    return {"deleted": name, "runs_removed": cur.rowcount, "artifex": artifex}


def poll_run(run_id: int, client: ArtifexClient | None = None) -> dict:
    """Proxy Artifex job state into training_runs; returns merged status."""
    client = _get_client(client)
    with conn() as db:
        run = db.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            raise RuntimeError(f"no run {run_id}")
        job_id = json.loads(run["config_json"]).get("job_id")
        status = client.train_status(job_id)
        state = status.get("state")
        if state == "completed" and run["status"] != "done":
            db.execute(
                "UPDATE training_runs SET status='done', finished_at=datetime('now'),"
                " artifact_path=? WHERE id=?",
                (status.get("lora"), run_id),
            )
        elif state in ("failed", "cancelled") and run["status"] == "running":
            db.execute(
                "UPDATE training_runs SET status = ?, finished_at = datetime('now')"
                " WHERE id = ?",
                (state, run_id),
            )
    return {"run_id": run_id, "job": status}


def default_probe_prompts(limit: int = 6) -> list[str]:
    """The user's own most-liked distinct prompts — a personal eval benchmark.

    Falls back to the highest-taste-scored prompts overall when liked images
    carry no prompts (the v1 migration stripped metadata from hand labels)."""
    liked_filter = """JOIN current_labels c ON c.image_id = i.id
                      AND c.kind='binary' AND c.value=1.0"""
    with conn() as db:
        for extra in (liked_filter, ""):
            rows = db.execute(
                f"""SELECT DISTINCT i.prompt FROM images i {extra}
                    WHERE i.prompt IS NOT NULL AND length(i.prompt) > 20
                    ORDER BY (SELECT score FROM predictions p WHERE p.image_id = i.id
                              AND p.model LIKE 'taste:%') DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()
            if rows:
                return [r["prompt"] for r in rows]
    return []


def eval_lora(run_id: int, lora_name: str | None = None,
              prompts: list[str] | None = None, seeds_per_prompt: int = 2,
              model: str | None = None, client: ArtifexClient | None = None) -> dict:
    """Probe-prompt evaluation of a trained LoRA vs. the bare checkpoint.

    Metrics per arm: taste (head), adherence (SigLIP text-image cosine),
    fidelity (cosine to the training-set centroid), diversity (1 - mean
    pairwise sim across seeds). Rows land in eval_results."""
    _assert_artifex_free()
    client = _get_client(client)
    head = latest_head()
    if head is None:
        raise RuntimeError("no taste head trained")
    prompts = prompts or default_probe_prompts()
    if not prompts:
        raise RuntimeError("no probe prompts available")

    with conn() as db:
        run = db.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            raise RuntimeError(f"no run {run_id}")
        fp = json.loads(run["dataset_fingerprint_json"] or "{}")
        train_ids = fp.get("image_ids", [])
        _, train_mat = load_vectors(db, MODEL_NAME, image_ids=train_ids)
        centroid = train_mat.mean(axis=0) if len(train_mat) else None
        if centroid is not None:
            centroid /= np.linalg.norm(centroid)
    lora_name = lora_name or run["name"]

    arms = {"base": None, "lora": [{"name": lora_name, "weight": DEFAULT_LORA_WEIGHT}]}
    summary = {}
    with conn() as db:
        for arm, loras in arms.items():
            taste_all, adhere_all, fid_all, div_all = [], [], [], []
            for prompt in prompts:
                pils, vecs_per_seed = [], []
                for s in range(seeds_per_prompt):
                    seed = 7_000_000 + s  # fixed seeds: comparable across arms/runs
                    payload = {"prompt": f"{prompt} --seed {seed}", "size": "832x1216"}
                    if model:
                        payload["model"] = model
                    if loras:
                        payload["loras"] = loras
                    out = client.generate(**payload)
                    data = base64.b64decode(out["data"][0]["b64_json"])
                    pils.append(Image.open(BytesIO(data)))
                scores, vecs = score_images(pils, head, return_vecs=True)
                tvec = text_features([prompt])[0]
                adherence = (vecs @ tvec).tolist()
                fidelity = (vecs @ centroid).tolist() if centroid is not None else [None] * len(vecs)
                sims = vecs @ vecs.T
                diversity = 1.0 - float(sims[np.triu_indices(len(vecs), 1)].mean()) \
                    if len(vecs) > 1 else None
                for j, (sc, ad, fi) in enumerate(zip(scores, adherence, fidelity)):
                    db.execute(
                        """INSERT INTO eval_results (run_id, checkpoint_step, prompt, seed,
                               taste_score, clip_score, fidelity_score, diversity_score)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (run_id, None, f"[{arm}] {prompt}", str(7_000_000 + j),
                         float(sc), float(ad), float(fi) if fi is not None else None,
                         diversity),
                    )
                taste_all += scores
                adhere_all += adherence
                fid_all += [f for f in fidelity if f is not None]
                if diversity is not None:
                    div_all.append(diversity)
            summary[arm] = {
                "taste": round(float(np.mean(taste_all)), 4),
                "adherence": round(float(np.mean(adhere_all)), 4),
                "fidelity": round(float(np.mean(fid_all)), 4) if fid_all else None,
                "diversity": round(float(np.mean(div_all)), 4) if div_all else None,
            }
        db.commit()
    return {"run_id": run_id, "prompts": len(prompts),
            "seeds_per_prompt": seeds_per_prompt, "summary": summary}
