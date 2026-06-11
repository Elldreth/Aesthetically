"""Thin client for the Artifex SDXL sidecar (https://localhost:7860 by default).

Phase 1 stub: connectivity + the three endpoints later phases build on.
Artifex is self-contained; everything here is plain HTTP and optional —
the rater works fine with Artifex offline.
"""
from __future__ import annotations

import os

import httpx

ARTIFEX_URL = os.environ.get("ARTIFEX_URL", "http://127.0.0.1:7860")


class ArtifexClient:
    def __init__(self, base_url: str = ARTIFEX_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def is_up(self) -> bool:
        try:
            return self._client.get("/openapi.json").status_code == 200
        except httpx.HTTPError:
            return False

    def generate(self, prompt: str, **kwargs) -> dict:
        """POST /v1/images/generations (OpenAI-Images-style contract)."""
        payload = {"prompt": prompt, **kwargs}
        r = self._client.post("/v1/images/generations", json=payload, timeout=None)
        r.raise_for_status()
        return r.json()

    def analyze_dataset(self, images: list[str], captions: list[str] | None = None,
                        face: bool = True) -> dict:
        """POST /v1/dataset/analyze — CLIP outliers/dupes/blur/caption-mismatch."""
        payload = {"images": images, "face": face}
        if captions is not None:
            payload["captions"] = captions
        r = self._client.post("/v1/dataset/analyze", json=payload, timeout=None)
        r.raise_for_status()
        return r.json()

    def train(self, images: list[str], name: str, **kwargs) -> dict:
        """POST /v1/train — start a LoRA training job; poll with train_status()."""
        payload = {"images": images, "name": name, **kwargs}
        r = self._client.post("/v1/train", json=payload, timeout=None)
        r.raise_for_status()
        return r.json()

    def train_status(self, job_id: str) -> dict:
        r = self._client.get(f"/v1/train/{job_id}")
        r.raise_for_status()
        return r.json()
