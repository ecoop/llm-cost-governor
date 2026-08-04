# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Shared base for backend-backed rolling-window counters.

The Tier-1 spend circuit breaker and the cumulative upload-byte cap both
need the same machinery: an in-memory counter that is the source of
truth between writes, persisted to a small blob by a debounced
background writer thread so the totals survive an ephemeral instance's
restart. This module factors that machinery out so both counters share
it.

What the base owns:
    - the state lock (subclasses guard their own dicts with ``self._lock``)
    - the persistence lifecycle: ``load`` / ``shutdown`` / ``flush``
    - the background writer thread that coalesces dirty signals into
      debounced blob rewrites

What each subclass provides:
    - ``to_dict`` / ``from_dict`` — its own state serialization
    - ``_read_blob`` / ``_write_blob`` — I/O for its object. Concrete
      subclasses hold a ``StateBackend`` + ``object_name`` as instance
      attributes and delegate to ``self._backend.read(self._object_name)``
      / ``.write(...)``. Tests inject an in-memory backend at
      construction time.
    - optionally ``_write_failure_alert`` — the operator alert tuple
      to emit when a write fails (``None`` → log only).

Persistence binds only when the caller passes ``enabled=True`` (via
``load``). Off-persistence the counter still records in memory for
local visibility but never writes — safe for local dev / CI without
any state backend.

Single-instance assumption: correct only at ``max-instances=1`` —
there is no cross-instance atomic increment. Multi-instance
correctness is out of scope for this base.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from llm_governor.alerts import alert
from llm_governor.state import StateBackend

# How long the writer thread waits after a record before flushing, so a burst
# of records coalesces into a single blob rewrite.
_WRITE_DEBOUNCE_SECONDS = 2.0


class GcsBackedCounter:
    """Base class: lock + persistence lifecycle + debounced writer thread.

    Subclasses hold their own state dicts (guarded by ``self._lock``) and
    implement ``to_dict`` / ``from_dict`` / ``_read_blob`` / ``_write_blob``.
    Thread-safe: ``record`` runs on the sync-endpoint threadpool while the
    writer thread reads snapshots concurrently; all state mutation and
    snapshotting happens under the single lock.

    Name is historical — since #299 the class routes through the
    pluggable ``state`` backend, so it isn't GCS-specific. Rename planned
    in a later extraction commit.
    """

    def __init__(
        self,
        *,
        name: str,
        object_name: str,
        writer_thread_name: str,
        log: logging.Logger,
        debounce_seconds: float = _WRITE_DEBOUNCE_SECONDS,
    ) -> None:
        self._name = name
        self._object_name = object_name
        self._writer_thread_name = writer_thread_name
        self._log = log
        self._debounce = debounce_seconds

        self._lock = threading.Lock()
        self._persist = False               # set true by load(enabled=True)
        self._backend_kind = ""             # informational; set by load()
        self._dirty = threading.Event()
        self._stop = threading.Event()
        self._writer: threading.Thread | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def load(self, *, enabled: bool = False, backend_kind: str = "") -> None:
        """Initialize state at startup; read backend + start writer when enabled.

        Safe to call when the backend is unreachable — a read failure
        logs and falls back to empty state so the app still boots (with
        caps binding from zero). A missing blob is normal on first run.

        Args:
            enabled: Whether persistence is active. When False, the
                counter runs in memory only (local visibility, no
                writes). Typically wired from the app's ``demo_mode``
                gate at startup.
            backend_kind: Informational label logged on successful
                load, e.g. ``"local"`` or ``"gcs"``. Purely cosmetic
                — no behavioral effect.
        """
        self._persist = enabled
        self._backend_kind = backend_kind
        if not self._persist:
            self._log.info("%s: persistence disabled — in-memory only.", self._name)
            return

        try:
            raw = self._read_blob()
        except Exception:  # noqa: BLE001 — never let backend trouble block startup
            self._log.exception("%s: failed reading state blob; starting empty.", self._name)
            raw = None

        if raw is None:
            self._log.info("%s: no state blob found; starting at zero.", self._name)
        else:
            try:
                self.from_dict(json.loads(raw))
                self._log.info(
                    "%s: loaded state for %s (backend: %s).",
                    self._name, self._object_name, backend_kind or "unspecified",
                )
            except Exception:  # noqa: BLE001
                self._log.exception("%s: malformed state blob; starting empty.", self._name)

        self._start_writer()

    def shutdown(self) -> None:
        """Stop the writer thread after a final synchronous flush (best effort)."""
        if self._writer is not None:
            self.flush()
            self._stop.set()
            self._dirty.set()  # wake the loop so it can observe _stop

    def flush(self) -> None:
        """Synchronously write the current snapshot to the backend (best effort)."""
        if not self._persist:
            return
        try:
            self._write_blob(json.dumps(self.to_dict()))
        except Exception:  # noqa: BLE001 — a failed write must never break a response
            self._log.exception("%s: state write failed; state not persisted.", self._name)
            alert_tuple = self._write_failure_alert()
            if alert_tuple is not None:
                # alert() is best-effort by contract — it catches and logs any
                # sink failure so the write-failure log line above isn't masked.
                alert(*alert_tuple)

    # ── writer thread ──────────────────────────────────────────────────────────

    def _start_writer(self) -> None:
        if self._writer is not None:
            return
        self._writer = threading.Thread(
            target=self._writer_loop, name=self._writer_thread_name, daemon=True
        )
        self._writer.start()

    def _schedule_write(self) -> None:
        if self._persist:
            self._dirty.set()

    def _writer_loop(self) -> None:
        """Coalesce dirty signals into debounced blob rewrites."""
        while not self._stop.is_set():
            self._dirty.wait()
            if self._stop.is_set():
                break
            self._dirty.clear()
            # Debounce: let a burst of records accumulate into one write.
            time.sleep(self._debounce)
            self.flush()

    # ── subclass hooks ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize current state to the persisted-blob schema. Override."""
        raise NotImplementedError

    def from_dict(self, data: dict) -> None:
        """Replace in-memory state from a deserialized blob. Override."""
        raise NotImplementedError

    def _read_blob(self) -> str | None:
        """Return the state blob's text, or None if absent. Override.

        Concrete subclasses typically hold a StateBackend + object_name
        as instance attributes and implement this as ``return
        self._backend.read(self._object_name)``. Tests substitute an
        in-memory backend at construction time.
        """
        raise NotImplementedError

    def _write_blob(self, text: str) -> None:
        """Overwrite the state blob with ``text``. Override.

        Concrete subclasses hold a StateBackend + object_name as
        instance attributes and implement this as ``self._backend.write(
        self._object_name, text)``.
        """
        raise NotImplementedError

    def _write_failure_alert(self) -> tuple[str, str, str] | None:
        """Return a ``(level, title, body)`` alert tuple on write failure.

        Default: ``None`` (log only). Subclasses override to notify the
        operator that persistence is failing.
        """
        return None


# ── State I/O helpers ─────────────────────────────────────────────────────────
#
# Thin convenience wrappers around a caller-supplied ``StateBackend``.
# Callers can hit the backend directly; these exist mainly for symmetry
# with the counter's own ``_read_blob`` / ``_write_blob`` method shape
# and as a stable helper import surface for tests.


def read_blob(backend: StateBackend, object_name: str) -> str | None:
    """Return ``object_name``'s text from ``backend``, or None if absent."""
    return backend.read(object_name)


def write_blob(backend: StateBackend, object_name: str, text: str) -> None:
    """Overwrite ``object_name`` in ``backend`` with ``text``."""
    backend.write(object_name, text)
