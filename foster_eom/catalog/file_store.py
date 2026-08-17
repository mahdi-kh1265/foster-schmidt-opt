"""Content-addressed immutable file store (Prompt 08).

Files are stored at ``models/{sha256[:2]}/{sha256}.{ext}`` relative to the
library database. The store is append-only: once written, content is never
modified or deleted.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


class ContentAddressedStore:
    """Immutable content-addressed file store keyed by SHA-256.

    Directory layout::

        root/
        ├── ab/
        │   └── ab3f…d7e.s1p
        └── c1/
            └── c1a2…f93.s2p
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _blob_path(self, sha256: str, ext: str) -> Path:
        """Return the storage path for a given SHA and extension."""
        prefix = sha256[:2]
        return self.root / prefix / f"{sha256}{ext}"

    def store(self, source_path: Path) -> str:
        """Store a file. Returns its SHA-256 hex digest.

        If a file with the same SHA already exists, this is a no-op
        (content-addressed deduplication).
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        sha = compute_sha256(source)
        ext = source.suffix.lower()
        dest = self._blob_path(sha, ext)

        if dest.exists():
            return sha  # idempotent no-op

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(dest))
        return sha

    def store_bytes(self, data: bytes, ext: str) -> str:
        """Store raw bytes. Returns SHA-256 hex digest."""
        sha = compute_bytes_sha256(data)
        if not ext.startswith("."):
            ext = f".{ext}"
        dest = self._blob_path(sha, ext)
        if dest.exists():
            return sha
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return sha

    def retrieve(self, sha256: str, ext: str) -> Path:
        """Return the path to a stored file.

        Raises FileNotFoundError if not present.
        """
        if not ext.startswith("."):
            ext = f".{ext}"
        p = self._blob_path(sha256, ext)
        if not p.exists():
            raise FileNotFoundError(f"No file in store with SHA {sha256[:12]}…")
        return p

    def verify(self, sha256: str, ext: str) -> bool:
        """Re-hash the stored file and compare to expected SHA-256."""
        p = self.retrieve(sha256, ext)
        return compute_sha256(p) == sha256

    def contains(self, sha256: str, ext: str) -> bool:
        """Check whether a file with the given SHA exists."""
        if not ext.startswith("."):
            ext = f".{ext}"
        return self._blob_path(sha256, ext).exists()
