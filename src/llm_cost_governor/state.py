# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Pluggable durable-state backend.

Decouples counters/state facades from any specific storage provider.
Ships with a local-file backend (default for laptops / CI) and a
lazily-imported GCS backend (for hosted deployments). Additional
providers (S3, Azure Blob, Redis) can be added as new StateBackend
classes plus a new value in ``get_backend`` with zero caller changes.

Zero coupling to any host application — every value the backends need
(bucket name, file-system root) is a constructor arg. Callers construct
what they want, either directly (``LocalFileBackend(some_path)``) or
through the ``get_backend(kind, ...)`` convenience factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StateBackend(Protocol):
    """Moves named JSON blobs to and from durable storage.

    Deliberately content-agnostic — callers own the schema, serialization,
    and any DEMO_MODE-style gating. A backend only reads/writes bytes by
    object name, so any object store (local disk, GCS, S3, Azure Blob)
    can satisfy it without the callers knowing which.
    """

    def read(self, object_name: str) -> str | None:
        """Return the named object's text, or ``None`` if it doesn't exist."""
        ...

    def write(self, object_name: str, text: str) -> None:
        """Overwrite the named object with ``text``."""
        ...


class LocalFileBackend:
    """Store each state object as a JSON file under a local directory.

    No external deps, creds, or network — the local-dev / CI provider.
    The directory is caller-supplied; the backend will create it on
    first write.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, object_name: str) -> Path:
        return self._root / object_name

    def read(self, object_name: str) -> str | None:
        try:
            return self._path(object_name).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def write(self, object_name: str, text: str) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then atomically replace, so a crash mid-write
        # never leaves a truncated blob that would fail to parse on reload.
        tmp = self._path(object_name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self._path(object_name))


class GcsBackend:
    """Store each state object as a blob in one GCS bucket.

    The ``google-cloud-storage`` client is imported lazily and cached
    per instance, so importing this module never requires the library
    — only an actual read/write does.
    """

    def __init__(self, bucket: str) -> None:
        self._bucket_name = bucket
        self._client = None

    def _bucket(self):
        if self._client is None:
            from google.cloud import storage  # lazy: only when GCS is used
            self._client = storage.Client()
        return self._client.bucket(self._bucket_name)

    def read(self, object_name: str) -> str | None:
        blob = self._bucket().blob(object_name)
        if not blob.exists():
            return None
        return blob.download_as_text()

    def write(self, object_name: str, text: str) -> None:
        blob = self._bucket().blob(object_name)
        blob.upload_from_string(text, content_type="application/json")


def get_backend(
    kind: str,
    *,
    data_dir: Path | None = None,
    gcs_bucket: str | None = None,
) -> StateBackend:
    """Construct a backend by name.

    Convenience factory equivalent to instantiating the concrete class
    directly — callers who already know they want a specific backend
    can skip this and construct e.g. ``LocalFileBackend(some_path)``.

    Args:
        kind: Backend selector, ``"local"`` or ``"gcs"``.
        data_dir: Filesystem directory for ``"local"``. Required when
            ``kind == "local"``. The backend creates the directory on
            first write.
        gcs_bucket: Bucket name for ``"gcs"``. Required when
            ``kind == "gcs"``.

    Raises:
        ValueError: When ``kind`` isn't a known backend, or when the
            arg the chosen backend needs is missing.
    """
    if kind == "local":
        if data_dir is None:
            raise ValueError("get_backend(kind='local') requires data_dir=...")
        return LocalFileBackend(data_dir)
    if kind == "gcs":
        if gcs_bucket is None:
            raise ValueError("get_backend(kind='gcs') requires gcs_bucket=...")
        return GcsBackend(gcs_bucket)
    raise ValueError(
        f"Unknown state backend kind {kind!r}; expected 'local' or 'gcs'."
    )
