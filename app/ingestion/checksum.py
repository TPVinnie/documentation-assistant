"""Content hashing used for duplicate detection and change detection (FR-01, FR-03)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
