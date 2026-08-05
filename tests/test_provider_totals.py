# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Tests for the per-provider totals read-model and its hook.

Covers:
    - ProviderTotalsHook: post accumulates cost per provider; pre is a
      no-op; it satisfies the runtime-checkable ``Hook`` Protocol; the
      default constructor gives a working in-memory instance.
    - ProviderTotals: accumulation across calls, ``unknown`` fallback for
      a missing provider, snapshot is a defensive copy, ``total`` sums.
    - Persistence: opt-in via a StateBackend — a round-trip through
      ``flush`` + a fresh instance restores totals; in-memory-only (no
      backend) records fine and never writes; ``enabled=True`` without a
      backend is rejected at construction.

No google-cloud-storage dependency or network is needed — persistence is
exercised through a tiny in-memory StateBackend.
"""

from __future__ import annotations

import pytest

from llm_cost_governor.provider_totals import ProviderTotals, ProviderTotalsHook
from llm_cost_governor.schemas import TokenEstimate, UsageRecord
from llm_cost_governor.wrapper import CallContext, Hook

# ── Test doubles / helpers ────────────────────────────────────────────────────


class _InMemoryBackend:
    """A StateBackend (duck-typed) backed by a plain dict — no disk, no GCS."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def read(self, object_name: str) -> str | None:
        return self._d.get(object_name)

    def write(self, object_name: str, text: str) -> None:
        self._d[object_name] = text


def _ctx() -> CallContext:
    """A minimal CallContext — the hook ignores it, but post() takes one."""
    return CallContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        kwargs={},
        tags={},
        estimate=TokenEstimate(input_tokens=0, output_tokens=0),
    )


def _usage(provider: str, cost_usd: float) -> UsageRecord:
    return UsageRecord(
        provider=provider,
        model="a-model",
        input_tokens=0,
        output_tokens=0,
        cost_usd=cost_usd,
    )


# ── ProviderTotalsHook ────────────────────────────────────────────────────────


def test_hook_post_accumulates_cost_per_provider():
    hook = ProviderTotalsHook()

    hook.post(_ctx(), _usage("anthropic", 0.50))
    hook.post(_ctx(), _usage("voyage", 0.20))
    hook.post(_ctx(), _usage("anthropic", 0.05))

    assert hook.totals.snapshot() == {"anthropic": 0.55, "voyage": 0.20}


def test_hook_pre_is_noop():
    hook = ProviderTotalsHook()
    assert hook.pre(_ctx()) is None
    assert hook.totals.snapshot() == {}


def test_hook_satisfies_hook_protocol():
    # runtime_checkable duck typing: name + pre + post is all it takes.
    assert isinstance(ProviderTotalsHook(), Hook)


def test_hook_default_constructs_working_in_memory_totals():
    hook = ProviderTotalsHook()
    assert isinstance(hook.totals, ProviderTotals)
    hook.post(_ctx(), _usage("anthropic", 1.0))
    assert hook.totals.snapshot() == {"anthropic": 1.0}


def test_hook_shares_a_passed_totals_instance():
    totals = ProviderTotals()
    ProviderTotalsHook(totals).post(_ctx(), _usage("anthropic", 0.3))
    # The externally-held instance sees the write — the point of passing one.
    assert totals.snapshot() == {"anthropic": 0.3}


# ── ProviderTotals accumulation ───────────────────────────────────────────────


def test_record_accumulates_and_returns_amount():
    totals = ProviderTotals()
    assert totals.record("anthropic", 0.10) == pytest.approx(0.10)
    totals.record("anthropic", 0.25)
    assert totals.snapshot() == {"anthropic": pytest.approx(0.35)}


def test_missing_provider_folds_into_unknown():
    totals = ProviderTotals()
    totals.record(None, 0.10)
    totals.record("", 0.05)
    assert totals.snapshot() == {"unknown": pytest.approx(0.15)}


def test_none_cost_is_treated_as_zero():
    totals = ProviderTotals()
    assert totals.record("anthropic", None) == 0.0  # type: ignore[arg-type]
    assert totals.snapshot() == {"anthropic": 0.0}


def test_snapshot_is_a_defensive_copy():
    totals = ProviderTotals()
    totals.record("anthropic", 1.0)
    snap = totals.snapshot()
    snap["anthropic"] = 999.0
    snap["injected"] = 5.0
    # Mutating the returned dict must not touch internal state.
    assert totals.snapshot() == {"anthropic": 1.0}


def test_total_sums_across_providers():
    totals = ProviderTotals()
    totals.record("anthropic", 0.50)
    totals.record("voyage", 0.20)
    totals.record("openai", 0.05)
    assert totals.total() == pytest.approx(0.75)


# ── Persistence (opt-in) ──────────────────────────────────────────────────────


def test_enabled_requires_a_backend():
    with pytest.raises(ValueError, match="requires a backend"):
        ProviderTotals(enabled=True)


def test_in_memory_default_never_writes():
    totals = ProviderTotals()
    totals.load()
    totals.record("anthropic", 0.10)
    totals.flush()  # no backend, persistence off → must be a harmless no-op
    assert totals.snapshot() == {"anthropic": pytest.approx(0.10)}


def test_persistence_round_trip_restores_totals():
    backend = _InMemoryBackend()

    first = ProviderTotals(backend=backend, enabled=True, object_name="pt.json")
    first.load()
    first.record("anthropic", 0.50)
    first.record("voyage", 0.20)
    first.flush()  # synchronous write, no dependence on the writer thread
    first.shutdown()

    # A fresh "process" over the same backend restores the prior totals.
    second = ProviderTotals(backend=backend, enabled=True, object_name="pt.json")
    second.load()
    assert second.snapshot() == {
        "anthropic": pytest.approx(0.50),
        "voyage": pytest.approx(0.20),
    }
    second.shutdown()


def test_missing_blob_starts_empty():
    second = ProviderTotals(backend=_InMemoryBackend(), enabled=True)
    second.load()  # nothing persisted yet
    assert second.snapshot() == {}
    second.shutdown()
