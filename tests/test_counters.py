# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Tests for the enforcement counters (llm_cost_governor.counters).

Covers the layer that had no dedicated suite before:

    RollingWeekCounter (the generic per-key rolling-week cap counter):
        - strict enforcement blocks the crossing event (current+incoming>cap)
        - lenient enforcement tolerates a one-event overshoot (current>=cap)
        - cap id + declaration-order precedence across dimensions
        - dimensions are independent (token vs ip)
        - weekly rollover resets a key's tally
        - record increments count / returns amount; snapshot_for privacy
        - to_dict/from_dict round-trip; _trim drops oldest by last_seen
        - disabled gate: enforce no-op, no writes
        - persistence: load/record/flush through an in-memory backend
        - unknown-dimension kwargs and bad construction raise

    CostCounter + WindowedCapHook (the spend counter and its guarded_call hook):
        - lenient one-call overshoot; tightest-window-first cap precedence
        - per-token weekly window; hourly rollover
        - cap-reached alerts latch once per window
        - snapshot_for privacy boundary
        - hook pre() raises on cap, post() records; identity_provider=None

Time is controlled by monkeypatching the module-level bucket helpers
(_week_of / _this_hour / _today), and spend pricing by patching
counters.usd_for_usage, so no test depends on the wall clock or the
pricing tables. Counters use an in-memory backend injected at
construction, matching the test_state_backends style.
"""

from __future__ import annotations

import pytest

# Reach through the module so monkeypatch.setattr(counters, "_week_of", ...) /
# (counters, "alert", ...) rebinds the names the methods actually look up.
from llm_cost_governor import counters
from llm_cost_governor.counters import (
    CapDimension,
    CostCapExceeded,
    CostCounter,
    RollingWeekCounter,
    WindowedCapHook,
)

# ── helpers ───────────────────────────────────────────────────────────────────


class InMemoryBackend:
    """StateBackend duck-type: an in-process object store (see test_state_backends)."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def read(self, object_name: str) -> str | None:
        return self._d.get(object_name)

    def write(self, object_name: str, text: str) -> None:
        self._d[object_name] = text


def _upload_counter(backend=None, *, enabled=False, strict=True, cap=1000):
    """A two-dimension (token, ip) strict byte counter — the upload use case."""
    return RollingWeekCounter(
        name="upload_counter",
        object_name="upload_counter.json",
        backend=backend or InMemoryBackend(),
        enabled=enabled,
        strict=strict,
        dimensions=[
            CapDimension("token", cap, "upload_token"),
            CapDimension("ip", cap, "upload_ip"),
        ],
    )


# ── RollingWeekCounter: strict enforcement ─────────────────────────────────────


def test_strict_allows_up_to_cap():
    c = _upload_counter(enabled=True, cap=1000)
    c.record(1000, token="t1")  # exactly at cap is allowed to land
    # A zero-byte follow-up doesn't cross (current + 0 == cap, not > cap).
    c.enforce(0, token="t1")


def test_strict_blocks_the_crossing_event_not_a_smaller_one():
    c = _upload_counter(enabled=True, cap=1000)
    c.record(900, token="t1")
    c.enforce(100, token="t1")  # 900 + 100 == 1000, not over → allowed
    with pytest.raises(CostCapExceeded) as ei:
        c.enforce(101, token="t1")  # 900 + 101 > 1000 → blocked
    assert ei.value.cap == "upload_token"


def test_strict_enforce_is_preflight_only_current_unchanged():
    c = _upload_counter(enabled=True, cap=1000)
    c.record(500, token="t1")
    with pytest.raises(CostCapExceeded):
        c.enforce(600, token="t1")
    # A rejected enforce must not have recorded anything.
    assert c.snapshot_for(token="t1")["caller_weekly"]["token"] == 500


# ── RollingWeekCounter: lenient enforcement ────────────────────────────────────


def test_lenient_tolerates_one_overshoot_then_blocks():
    c = _upload_counter(enabled=True, strict=False, cap=1.0)
    c.enforce(token="t1")            # nothing recorded yet → allowed
    c.record(1.5, token="t1")        # single event overshoots the cap
    with pytest.raises(CostCapExceeded) as ei:
        c.enforce(token="t1")        # now current (1.5) >= cap → blocked
    assert ei.value.cap == "upload_token"


# ── RollingWeekCounter: dimensions ─────────────────────────────────────────────


def test_dimension_precedence_is_declaration_order():
    c = _upload_counter(enabled=True, cap=100)
    c.record(100, token="t1", ip="1.2.3.4")  # both dimensions now at cap
    with pytest.raises(CostCapExceeded) as ei:
        c.enforce(1, token="t1", ip="1.2.3.4")
    assert ei.value.cap == "upload_token"  # token declared first → named first


def test_dimensions_are_independent():
    c = _upload_counter(enabled=True, cap=100)
    c.record(100, token="t1", ip="1.2.3.4")
    # A different token on a different ip is unaffected.
    c.enforce(50, token="t2", ip="5.6.7.8")
    # Same ip but no token key → only the ip dimension is checked, and it's full.
    with pytest.raises(CostCapExceeded) as ei:
        c.enforce(1, ip="1.2.3.4")
    assert ei.value.cap == "upload_ip"


def test_absent_key_dimension_is_skipped():
    c = _upload_counter(enabled=True, cap=100)
    c.record(100, ip="1.2.3.4")  # ip dimension is now full
    # ip key omitted → ip dimension not checked; token has room → allowed.
    c.enforce(1, token="t1")


# ── RollingWeekCounter: weekly rollover ────────────────────────────────────────


def test_weekly_rollover_resets_key(monkeypatch):
    c = _upload_counter(enabled=True, cap=1000)

    monkeypatch.setattr(counters, "_week_of", lambda: "2026-08-03")
    c.record(1000, token="t1")
    with pytest.raises(CostCapExceeded):
        c.enforce(1, token="t1")

    # New ISO week → the token's weekly allowance is fresh.
    monkeypatch.setattr(counters, "_week_of", lambda: "2026-08-10")
    c.enforce(1000, token="t1")
    assert c.snapshot_for(token="t1")["caller_weekly"]["token"] is None


# ── RollingWeekCounter: record / snapshot ──────────────────────────────────────


def test_record_returns_amount_and_counts_events():
    c = _upload_counter(cap=10_000)
    assert c.record(1200, token="t1") == 1200
    c.record(800, token="t1")
    stat = c._tables["token"]["t1"]
    assert stat.amount_cumulative_week == 2000
    assert stat.event_count == 2


def test_snapshot_for_is_privacy_scoped():
    c = _upload_counter(cap=5000)
    c.record(1000, token="t1", ip="1.2.3.4")
    c.record(2000, token="t2", ip="5.6.7.8")

    snap = c.snapshot_for(token="t1")
    assert snap["caps"] == {"token": 5000, "ip": 5000}
    assert snap["caller_weekly"]["token"] == 1000
    # ip not supplied → None; no other caller's data anywhere in the payload.
    assert snap["caller_weekly"]["ip"] is None
    assert "t2" not in repr(snap) and "5.6.7.8" not in repr(snap)


def test_snapshot_for_unknown_key_is_none():
    c = _upload_counter(cap=5000)
    assert c.snapshot_for(token="nope")["caller_weekly"]["token"] is None


# ── RollingWeekCounter: serialization / trim ───────────────────────────────────


def test_to_dict_from_dict_round_trip():
    c1 = _upload_counter(cap=5000)
    c1.record(1000, token="t1", ip="1.2.3.4")
    c1.record(2000, token="t2")
    blob = c1.to_dict()

    c2 = _upload_counter(cap=5000)
    c2.from_dict(blob)
    assert c2.snapshot_for(token="t1")["caller_weekly"]["token"] == 1000
    assert c2.snapshot_for(token="t2")["caller_weekly"]["token"] == 2000
    assert c2.to_dict() == blob


def test_from_dict_ignores_unknown_dimensions_and_tolerates_gaps():
    c = _upload_counter(cap=5000)
    c.from_dict(
        {
            "dimensions": {
                "token": {"t1": {"amount_cumulative_week": 42, "week_of": counters._week_of()}},
                "mystery": {"x": {"amount_cumulative_week": 9}},  # unknown dim → ignored
            }
        }
    )
    assert c.snapshot_for(token="t1")["caller_weekly"]["token"] == 42
    assert "mystery" not in c._tables


def test_trim_drops_oldest_by_last_seen():
    c = RollingWeekCounter(
        name="upload_counter",
        object_name="upload_counter.json",
        backend=InMemoryBackend(),
        dimensions=[CapDimension("token", 10_000, "upload_token", max_keys=2)],
    )
    # last_seen is set from _now_iso(); stub it so ordering is deterministic.
    for key, ts in [("old", "2026-08-01T00:00:00"), ("mid", "2026-08-02T00:00:00")]:
        c.record(1, token=key)
        c._tables["token"][key].last_seen = ts
    c.record(1, token="new")  # third key trips the max_keys=2 trim
    c._tables["token"]["new"].last_seen = "2026-08-03T00:00:00"
    c._trim()

    assert set(c._tables["token"]) == {"mid", "new"}  # "old" evicted


# ── RollingWeekCounter: gate / persistence / construction ──────────────────────


def test_disabled_gate_enforce_is_noop_and_no_writes():
    backend = InMemoryBackend()
    c = _upload_counter(backend, enabled=False, cap=100)
    c.load()
    c.record(1_000_000, token="t1")  # way over cap, but disabled
    c.enforce(1_000_000, token="t1")  # no raise: enforcement gated off
    c.flush()
    assert backend.read("upload_counter.json") is None  # nothing persisted


def test_persistence_round_trips_through_backend():
    backend = InMemoryBackend()
    c1 = _upload_counter(backend, enabled=True, cap=5000)
    c1.load()  # no blob yet → starts empty
    try:
        c1.record(1500, token="t1", ip="1.2.3.4")
        c1.flush()  # synchronous write; no need to wait on the debounce thread
    finally:
        c1.shutdown()
    assert backend.read("upload_counter.json") is not None

    c2 = _upload_counter(backend, enabled=True, cap=5000)
    c2.load()  # reads the blob written above
    try:
        assert c2.snapshot_for(token="t1")["caller_weekly"]["token"] == 1500
    finally:
        c2.shutdown()


def test_unknown_dimension_kwarg_raises():
    c = _upload_counter(cap=100)
    with pytest.raises(TypeError):
        c.record(1, tokne="t1")  # typo'd dimension name
    with pytest.raises(TypeError):
        c.enforce(1, user="t1")


def test_construction_validates_dimensions():
    with pytest.raises(ValueError):
        RollingWeekCounter(
            name="x", object_name="x.json", backend=InMemoryBackend(), dimensions=[]
        )
    with pytest.raises(ValueError):
        RollingWeekCounter(
            name="x",
            object_name="x.json",
            backend=InMemoryBackend(),
            dimensions=[
                CapDimension("token", 1, "a"),
                CapDimension("token", 2, "b"),  # duplicate name
            ],
        )


def test_dimension_message_builder_is_used():
    c = RollingWeekCounter(
        name="upload_counter",
        object_name="u.json",
        backend=InMemoryBackend(),
        enabled=True,
        dimensions=[
            CapDimension(
                "token",
                5_242_880,
                "upload_token",
                message=lambda cap: f"weekly upload allowance ({cap // 1024 // 1024} MiB) exceeded",
            )
        ],
    )
    c.record(5_242_880, token="t1")
    with pytest.raises(CostCapExceeded) as ei:
        c.enforce(1, token="t1")
    assert str(ei.value) == "weekly upload allowance (5 MiB) exceeded"


# ── CostCounter helpers ────────────────────────────────────────────────────────


def _cost_counter(monkeypatch, *, backend=None, enabled=False, **caps):
    """A CostCounter whose per-call price is taken from usage['usd'] (patched)."""
    monkeypatch.setattr(counters, "usd_for_usage", lambda usage: usage["usd"])
    defaults = {
        "hourly_cap_usd": 0.50,
        "daily_cap_usd": 2.00,
        "weekly_cap_usd": 10.00,
        "per_token_cap_usd": 1.00,
    }
    defaults.update(caps)
    return CostCounter(
        object_name="cost_counter.json",
        backend=backend or InMemoryBackend(),
        enabled=enabled,
        **defaults,
    )


# ── CostCounter: enforcement ───────────────────────────────────────────────────


def test_cost_counter_lenient_one_call_overshoot(monkeypatch):
    c = _cost_counter(monkeypatch, enabled=True, hourly_cap_usd=0.50)
    c.enforce(None)                     # nothing spent → allowed
    c.record({"usd": 0.60})             # a single call overshoots the hourly cap
    with pytest.raises(CostCapExceeded) as ei:
        c.enforce(None)                 # next call blocked
    assert ei.value.cap == "hourly"


def test_cost_counter_cap_precedence_is_tightest_window_first(monkeypatch):
    c = _cost_counter(monkeypatch, enabled=True)
    c.record({"usd": 100.0})            # trips hourly, daily, and weekly at once
    with pytest.raises(CostCapExceeded) as ei:
        c.enforce(None)
    assert ei.value.cap == "hourly"     # checked before daily/weekly


def test_cost_counter_per_token_weekly_window(monkeypatch):
    # Global windows set huge so only the per-token allowance can trip.
    c = _cost_counter(
        monkeypatch,
        enabled=True,
        hourly_cap_usd=1e9,
        daily_cap_usd=1e9,
        weekly_cap_usd=1e9,
        per_token_cap_usd=1.00,
    )
    c.record({"usd": 1.20}, token="invite-A")
    with pytest.raises(CostCapExceeded) as ei:
        c.enforce("invite-A")
    assert ei.value.cap == "per_token"
    c.enforce("invite-B")  # a different token is unaffected


def test_cost_counter_hourly_rollover_resets(monkeypatch):
    c = _cost_counter(monkeypatch, enabled=True, hourly_cap_usd=0.50)
    monkeypatch.setattr(counters, "_this_hour", lambda: "2026-08-04T10")
    c.record({"usd": 0.60})
    with pytest.raises(CostCapExceeded):
        c.enforce(None)
    monkeypatch.setattr(counters, "_this_hour", lambda: "2026-08-04T11")
    c.enforce(None)  # fresh hour → allowed


def test_cost_counter_disabled_enforce_is_noop(monkeypatch):
    c = _cost_counter(monkeypatch, enabled=False, hourly_cap_usd=0.01)
    c.record({"usd": 5.0})
    c.enforce(None)  # gated off → no raise


# ── CostCounter: alerts ────────────────────────────────────────────────────────


def test_cost_counter_alert_latches_once_per_window(monkeypatch):
    fired: list[tuple[str, str, str]] = []
    monkeypatch.setattr(counters, "alert", lambda level, title, body: fired.append((level, title, body)))
    # Only the daily cap is reachable; the rest are set out of range.
    c = _cost_counter(
        monkeypatch,
        enabled=True,
        hourly_cap_usd=1e9,
        daily_cap_usd=0.50,
        weekly_cap_usd=1e9,
        per_token_cap_usd=1e9,
    )
    try:
        c.record({"usd": 1.0})  # crosses daily → one CRITICAL alert
        c.record({"usd": 1.0})  # still over, but the daily latch has fired
    finally:
        c.shutdown()

    daily_alerts = [t for t in fired if t[0] == counters.CRITICAL]
    assert len(daily_alerts) == 1


# ── CostCounter: snapshot ──────────────────────────────────────────────────────


def test_cost_counter_snapshot_for_is_privacy_scoped(monkeypatch):
    c = _cost_counter(monkeypatch)
    c.record({"usd": 0.30}, token="invite-A")
    c.record({"usd": 0.70}, token="invite-B")

    snap = c.snapshot_for("invite-A")
    assert snap["caller_weekly_usd"] == pytest.approx(0.30)
    assert "invite-B" not in repr(snap)  # no other token leaks
    assert snap["caps"]["per_token_usd"] == 1.00


# ── WindowedCapHook ────────────────────────────────────────────────────────────


class _Usage:
    """Minimal UsageRecord stand-in (only the attrs the hook reads)."""

    model = "claude-opus-4-8"
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


def test_windowed_cap_hook_pre_raises_and_post_records(monkeypatch):
    c = _cost_counter(monkeypatch, enabled=True, hourly_cap_usd=0.50)
    monkeypatch.setattr(counters, "usd_for_usage", lambda usage: 0.60)
    hook = WindowedCapHook(c, identity_provider=lambda: "invite-A")

    hook.pre(ctx=None)                 # nothing spent yet → no raise
    hook.post(ctx=None, usage=_Usage())  # records 0.60, crossing the hourly cap
    with pytest.raises(CostCapExceeded) as ei:
        hook.pre(ctx=None)
    assert ei.value.cap == "hourly"


def test_windowed_cap_hook_default_identity_is_global_only(monkeypatch):
    # Default identity_provider → None: only global windows enforce, no per-token.
    c = _cost_counter(
        monkeypatch,
        enabled=True,
        hourly_cap_usd=1e9,
        daily_cap_usd=1e9,
        weekly_cap_usd=1e9,
        per_token_cap_usd=0.01,
    )
    monkeypatch.setattr(counters, "usd_for_usage", lambda usage: 5.0)
    hook = WindowedCapHook(c)  # no identity_provider

    hook.post(ctx=None, usage=_Usage())
    hook.pre(ctx=None)  # per-token cap is tiny, but identity is None → not checked
