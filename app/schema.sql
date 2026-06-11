-- Aesthetically schema
-- Labels are append-only events; "current" state is derived (see current_labels).
-- Files stay on disk; the DB stores identity (sha256), sources, labels, vectors.

CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  sha256 TEXT NOT NULL UNIQUE,
  phash TEXT,
  width INTEGER,
  height INTEGER,
  format TEXT,
  file_size INTEGER,
  prompt TEXT,
  negative_prompt TEXT,
  model_hash TEXT,
  seed TEXT,
  gen_params_raw TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- An image can be known from many places: local copies, URLs.
CREATE TABLE IF NOT EXISTS image_sources (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id),
  kind TEXT NOT NULL CHECK (kind IN ('local', 'url')),
  location TEXT NOT NULL,
  first_seen TEXT NOT NULL DEFAULT (datetime('now')),
  last_verified TEXT,
  UNIQUE (image_id, location)
);
CREATE INDEX IF NOT EXISTS idx_sources_image ON image_sources(image_id);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY,
  name TEXT,
  started_at TEXT NOT NULL DEFAULT (datetime('now')),
  note TEXT
);

-- Append-only. binary: value 1/0.5/0. scalar: any rating. pairwise: this image
-- beat opponent_image_id. exclude: value 1 = unusable sample (corrupt/off-topic),
-- distinct from "disliked".
CREATE TABLE IF NOT EXISTS labels (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id),
  kind TEXT NOT NULL CHECK (kind IN ('binary', 'scalar', 'pairwise', 'exclude')),
  value REAL,
  opponent_image_id INTEGER REFERENCES images(id),
  source TEXT NOT NULL DEFAULT 'manual',  -- manual | migration | model:<name>
  session_id INTEGER REFERENCES sessions(id),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_labels_image ON labels(image_id, kind, id);

-- Latest label per (image, kind) wins.
CREATE VIEW IF NOT EXISTS current_labels AS
  SELECT l.* FROM labels l
  WHERE l.id = (
    SELECT max(id) FROM labels
    WHERE image_id = l.image_id AND kind = l.kind
  );

CREATE TABLE IF NOT EXISTS embeddings (
  image_id INTEGER NOT NULL REFERENCES images(id),
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL,                      -- float32 little-endian
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (image_id, model)
);

-- Model scores, replaced wholesale on retrain (history lives in taste model files,
-- not here — unlike labels, predictions are cheap to regenerate).
CREATE TABLE IF NOT EXISTS predictions (
  image_id INTEGER NOT NULL REFERENCES images(id),
  model TEXT NOT NULL,                    -- e.g. 'taste:v3'
  score REAL NOT NULL,                    -- P(like), 0..1
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (image_id, model)
);

-- Near-duplicate pairs surfaced by dedupe.py (image_id > dup_of_image_id never holds;
-- canonical = lowest id in the connected component).
CREATE TABLE IF NOT EXISTS near_dups (
  image_id INTEGER NOT NULL REFERENCES images(id),
  canonical_id INTEGER NOT NULL REFERENCES images(id),
  method TEXT NOT NULL,                   -- 'phash' | 'embedding' | 'phash+embedding'
  score REAL,                             -- hamming distance or cosine similarity
  PRIMARY KEY (image_id)
);

-- Reproducible dataset snapshots: the query that defined them plus frozen membership.
CREATE TABLE IF NOT EXISTS exports (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  format TEXT NOT NULL,
  query_json TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS export_items (
  export_id INTEGER NOT NULL REFERENCES exports(id),
  image_id INTEGER NOT NULL REFERENCES images(id),
  label_value REAL,
  PRIMARY KEY (export_id, image_id)
);

-- LoRA training experiment log (Phase 4 consumes these; recorded from day one).
CREATE TABLE IF NOT EXISTS training_runs (
  id INTEGER PRIMARY KEY,
  name TEXT,
  export_id INTEGER REFERENCES exports(id),  -- dataset snapshot used
  config_json TEXT NOT NULL,                 -- full Artifex /v1/train payload (sans image bytes)
  dataset_fingerprint_json TEXT,             -- size, diversity, face coverage, dedup stats
  status TEXT NOT NULL DEFAULT 'pending',    -- pending | running | done | failed | cancelled
  artifact_path TEXT,                        -- saved LoRA file
  started_at TEXT,
  finished_at TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS eval_results (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES training_runs(id),
  checkpoint_step INTEGER,
  prompt TEXT,
  seed TEXT,
  image_id INTEGER REFERENCES images(id),
  taste_score REAL,
  clip_score REAL,
  fidelity_score REAL,
  diversity_score REAL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_eval_run ON eval_results(run_id);
