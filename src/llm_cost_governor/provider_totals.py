# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Per-provider cumulative USD totals — a read-model, plus its hook.

The rolling-window ``CostCounter`` (``llm_cost_governor.counters``) is an
*enforcement* layer: it sums spend into global time windows to trip caps,
and in doing so aggregates the provider away — a $0.30 hour is $0.30
whether it went to Anthropic, Voyage, or both. Some apps want the
breakdown back: "how much of this went to each provider?" for a usage
widget or a cost dashboard.

``ProviderTotals`` is that read-model. It keeps one running USD total per
provider name, accumulated for the life of the process (or the life of
the persisted blob). It is deliberately **not** a counter in the
enforcement sense — no caps, no time windows, no weekly reset. It only
answers "how much, per provider, so far".

Persistence is opt-in. With no ``backend`` it is pure in-memory and
resets on restart (the common case — a live widget that only needs
since-boot numbers). Pass a ``StateBackend`` and ``enabled=True`` and the
totals ride the same debounced background-writer machinery as the other
counters (``GcsBackedCounter``), surviving an ephemeral instance's
restart.

``ProviderTotalsHook`` is the ``guarded_call`` / ``record_usage`` adapter:
its ``post`` adds each completed call's ``cost_usd`` to its ``provider``'s
running total. ``pre`` is a no-op — there is nothing to enforce.
"""

from __future__ import annotations

import logging

from llm_cost_governor.persistence import GcsBackedCounter
from llm_cost_governor.schemas import UsageRecord
from llm_cost_governor.state import StateBackend
from llm_cost_governor.wrapper import CallContext

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Provider label used when a call's provider is missing/blank. Provider
# names ("anthropic", "voyage") are not credentials, so — unlike the
# per-token tables in counters.py — the totals map is safe to expose whole.
_UNKNOWN = "unknown"


class ProviderTotals(GcsBackedCounter):
    """Cumulative USD spend per provider, optionally persisted.

    A running ``{provider: usd}`` map. ``record`` adds to a provider's
    total; ``snapshot`` returns a copy of the whole map; ``total`` sums
    across providers. Unbounded in time — the total is a lifetime (or
    life-of-blob) cumulative, not a windowed one.

    The lock and the persistence lifecycle (load / flush / shutdown +
    the debounced writer thread) come from ``GcsBackedCounter``; this
    class supplies only the per-provider state and its serialization.

    Args:
        backend: Where to persist state. Any ``StateBackend``, or
            ``None`` (the default) for pure in-memory operation — totals
            reset on restart. Required when ``enabled`` is True.
        object_name: Filename/key for the persisted state blob.
        enabled: Master gate for persistence. When False (default) the
            map lives in memory only (still fully functional for a live
            widget); nothing is read or written. When True a ``backend``
            must be supplied, ``load()`` restores prior totals, and the
            background writer persists after records.
        writer_thread_name: Background-writer thread name.
        log: Logger (defaults to this module's logger).
        write_failure_alert: Optional ``(level, title, body)`` operator
            alert emitted when persisting fails (default: log only).

    Raises:
        ValueError: When ``enabled`` is True but no ``backend`` was given.
    """

    def __init__(
        self,
        *,
        backend: StateBackend | None = None,
        object_name: str = "provider_totals.json",
        enabled: bool = False,
        writer_thread_name: str = "provider-totals-writer",
        log: logging.Logger | None = None,
        write_failure_alert: tuple[str, str, str] | None = None,
    ) -> None:
        if enabled and backend is None:
            raise ValueError(
                "ProviderTotals(enabled=True) requires a backend to persist to; "
                "pass backend=... or leave enabled=False for in-memory-only totals."
            )
        super().__init__(
            name="provider_totals",
            object_name=object_name,
            writer_thread_name=writer_thread_name,
            log=log or _log,
        )
        self._backend = backend
        self._enabled = enabled
        self._write_failure_alert_tuple = write_failure_alert
        self._totals: dict[str, float] = {}

    def load(self, *, backend_kind: str = "") -> None:  # type: ignore[override]
        """Initialize state; read backend + start writer when ``enabled``.

        Forwards to :meth:`GcsBackedCounter.load` with the instance's own
        ``enabled`` gate, so consumers pass ``enabled`` once (at
        construction) rather than repeating it here.
        """
        super().load(enabled=self._enabled, backend_kind=backend_kind)

    # ── recording ──────────────────────────────────────────────────────────────

    def record(self, provider: str | None, usd: float) -> float:
        """Add ``usd`` to ``provider``'s running total.

        Runs in all modes (in-memory visibility); only schedules a
        persist write when ``enabled``. A missing/blank provider is
        folded into ``"unknown"`` so a mislabeled call still counts
        rather than vanishing.

        Args:
            provider: The provider name (e.g. ``"anthropic"``); ``None``
                or ``""`` is recorded under ``"unknown"``.
            usd: The USD cost to add. ``None`` is treated as ``0.0``.

        Returns:
            The amount added (a float), for call-site symmetry with the
            other counters' ``record``.
        """
        key = provider or _UNKNOWN
        amount = float(usd or 0.0)
        with self._lock:
            self._totals[key] = self._totals.get(key, 0.0) + amount
        self._schedule_write()
        return amount

    # ── snapshots / serialization ──────────────────────────────────────────────

    def snapshot(self) -> dict[str, float]:
        """A copy of the current per-provider totals (safe to hand to callers).

        Returns a fresh dict, so a reader (e.g. a ``/usage`` endpoint)
        can hold or mutate it without touching internal state. Values are
        the exact accumulated floats — round at display time.
        """
        with self._lock:
            return dict(self._totals)

    def total(self) -> float:
        """Sum across all providers — total USD recorded so far."""
        with self._lock:
            return sum(self._totals.values())

    def to_dict(self) -> dict:
        """Serialize current state to the persisted-blob schema."""
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "totals": {p: round(u, 6) for p, u in self._totals.items()},
            }

    def from_dict(self, data: dict) -> None:
        """Replace in-memory state from a deserialized blob (tolerant of gaps)."""
        with self._lock:
            self._totals = {
                str(p): float(u) for p, u in data.get("totals", {}).items()
            }

    # ── persistence hooks (GcsBackedCounter) ───────────────────────────────────

    def _read_blob(self) -> str | None:
        if self._backend is None:
            return None
        return self._backend.read(self._object_name)

    def _write_blob(self, text: str) -> None:
        if self._backend is not None:
            self._backend.write(self._object_name, text)

    def _write_failure_alert(self) -> tuple[str, str, str] | None:
        return self._write_failure_alert_tuple


# ── Hook adapter for guarded_call / record_usage ────────────────────────────────


class ProviderTotalsHook:
    """`Hook` adapter that accumulates per-provider USD totals.

    ``post`` adds the completed call's ``cost_usd`` to its ``provider``'s
    running total in a :class:`ProviderTotals`; ``pre`` is a no-op (there
    is nothing to enforce). Compose it alongside enforcement hooks in a
    ``guarded_call`` chain to get a per-provider spend breakdown that the
    windowed ``CostCounter`` aggregates away.

    Args:
        totals: The :class:`ProviderTotals` to accumulate into. Omit to
            get a fresh in-memory instance (read it back via
            ``hook.totals.snapshot()``); pass a persisted one to survive
            restarts and to share the map across several hook chains.
    """

    name = "provider_totals"

    def __init__(self, totals: ProviderTotals | None = None) -> None:
        self.totals = totals if totals is not None else ProviderTotals()

    def pre(self, ctx: CallContext) -> None:
        return None

    def post(self, ctx: CallContext, usage: UsageRecord) -> None:
        self.totals.record(usage.provider, usage.cost_usd)
