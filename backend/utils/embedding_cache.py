"""In-memory LRU cache for reference signature embeddings."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import Lock

import torch


class EmbeddingLRUCache:
    """Thread-safe LRU cache storing detached CPU embeddings."""

    def __init__(self, max_size: int = 256):
        self.max_size = max(1, int(max_size))
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def key_from_bytes(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def get(self, key: str, device: str = "cpu") -> torch.Tensor | None:
        with self._lock:
            value = self._store.get(key)
            if value is None:
                return None

            # Mark as recently used.
            self._store.move_to_end(key)

        return value.clone().to(device)

    def put(self, key: str, embedding: torch.Tensor) -> None:
        # Store a detached CPU copy so cache is device-agnostic and lightweight.
        safe_value = embedding.detach().cpu().clone()

        with self._lock:
            self._store[key] = safe_value
            self._store.move_to_end(key)

            while len(self._store) > self.max_size:
                self._store.popitem(last=False)
