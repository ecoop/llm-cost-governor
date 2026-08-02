# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Unit tests for the pluggable durable-state backend.

Covers:
    - LocalFileBackend: write/read round-trip, missing object → None,
      survival across a fresh backend instance (simulated restart), and
      the atomic temp-file write (no ``.tmp`` left behind).
    - get_backend(kind, ...): selects LocalFileBackend / GcsBackend by
      name; raises ValueError on unknown selector or missing required
      args for the chosen backend.
    - GcsBackend: read/write drive the blob API correctly, with the
      lazy google-cloud-storage client replaced by a fake — no real
      dependency.
    - Protocol decoupling: an unrelated class satisfying the read/write
      shape is accepted as a StateBackend (runtime_checkable), proving
      the seam is duck-typed and a future S3/Azure provider drops in
      cleanly.

No google-cloud-storage dependency or network is needed.
"""

from pathlib import Path

import pytest

from llm_guardrails.state import (
    GcsBackend,
    LocalFileBackend,
    StateBackend,
    get_backend,
)

# ── LocalFileBackend ─────────────────────────────────────────────────────────────


def test_local_round_trip(tmp_path: Path):
    backend = LocalFileBackend(tmp_path)
    backend.write("counter.json", '{"n": 1}')
    assert backend.read("counter.json") == '{"n": 1}'


def test_local_read_missing_returns_none(tmp_path: Path):
    backend = LocalFileBackend(tmp_path)
    assert backend.read("absent.json") is None


def test_local_write_creates_root(tmp_path: Path):
    # Root does not exist yet; write() must create it.
    root = tmp_path / "nested" / ".local-state"
    backend = LocalFileBackend(root)
    backend.write("x.json", "hi")
    assert (root / "x.json").read_text(encoding="utf-8") == "hi"


def test_local_overwrites_existing(tmp_path: Path):
    backend = LocalFileBackend(tmp_path)
    backend.write("c.json", "old")
    backend.write("c.json", "new")
    assert backend.read("c.json") == "new"


def test_local_survives_restart(tmp_path: Path):
    # A first "process" writes; a second fresh instance over the same dir reads.
    LocalFileBackend(tmp_path).write("c.json", "durable")
    assert LocalFileBackend(tmp_path).read("c.json") == "durable"


def test_local_write_leaves_no_temp_file(tmp_path: Path):
    backend = LocalFileBackend(tmp_path)
    backend.write("c.json", "data")
    # The atomic temp file must have been replaced, not left behind.
    assert list(tmp_path.iterdir()) == [tmp_path / "c.json"]


# ── get_backend factory ──────────────────────────────────────────────────────────


def test_get_backend_local(tmp_path: Path):
    backend = get_backend("local", data_dir=tmp_path / "state")
    assert isinstance(backend, LocalFileBackend)
    backend.write("c.json", "ok")
    assert (tmp_path / "state" / "c.json").read_text(encoding="utf-8") == "ok"


def test_get_backend_gcs():
    backend = get_backend("gcs", gcs_bucket="my-bucket")
    assert isinstance(backend, GcsBackend)
    assert backend._bucket_name == "my-bucket"


def test_get_backend_unknown_kind_raises():
    with pytest.raises(ValueError, match="azure"):
        get_backend("azure")


def test_get_backend_local_requires_data_dir():
    with pytest.raises(ValueError, match="data_dir"):
        get_backend("local")


def test_get_backend_gcs_requires_bucket():
    with pytest.raises(ValueError, match="gcs_bucket"):
        get_backend("gcs")


# ── GcsBackend (fake client, no google-cloud-storage) ────────────────────────────


class _FakeBlob:
    def __init__(self, store: dict, name: str):
        self._store = store
        self._name = name

    def exists(self) -> bool:
        return self._name in self._store

    def download_as_text(self) -> str:
        return self._store[self._name]

    def upload_from_string(self, text: str, content_type: str | None = None) -> None:
        self._store[self._name] = text


class _FakeBucket:
    def __init__(self, store: dict):
        self._store = store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._store, name)


class _FakeClient:
    def __init__(self, store: dict):
        self._store = store

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self._store)


@pytest.fixture
def gcs_backend():
    """A GcsBackend whose lazy client is pre-seeded with a fake (no import)."""
    backend = GcsBackend("test-bucket")
    store: dict[str, str] = {}
    backend._client = _FakeClient(store)  # bypass the lazy google.cloud import
    return backend, store


def test_gcs_read_missing_returns_none(gcs_backend):
    backend, _store = gcs_backend
    assert backend.read("absent.json") is None


def test_gcs_round_trip(gcs_backend):
    backend, store = gcs_backend
    backend.write("c.json", '{"n": 2}')
    assert store["c.json"] == '{"n": 2}'
    assert backend.read("c.json") == '{"n": 2}'


# ── Protocol decoupling ──────────────────────────────────────────────────────────


def test_arbitrary_class_satisfies_protocol():
    """A future provider needs only read/write — no inheritance, no import."""

    class InMemoryBackend:
        def __init__(self):
            self._d: dict[str, str] = {}

        def read(self, object_name: str):
            return self._d.get(object_name)

        def write(self, object_name: str, text: str) -> None:
            self._d[object_name] = text

    backend = InMemoryBackend()
    assert isinstance(backend, StateBackend)  # runtime_checkable duck typing
    backend.write("k", "v")
    assert backend.read("k") == "v"


def test_llm_guardrails_persistence_helpers_take_a_backend(tmp_path: Path):
    """llm_guardrails.persistence.read_blob / write_blob take a backend arg."""
    from llm_guardrails import persistence

    backend = LocalFileBackend(tmp_path)
    persistence.write_blob(backend, "b.json", "value")
    assert persistence.read_blob(backend, "b.json") == "value"
