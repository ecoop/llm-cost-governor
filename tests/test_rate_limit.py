# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Tests for the Tier 1 per-IP rate limiter (api.rate_limit, #249 slice 2).

Covers:
    - IPRateLimiter.check: allows up to the limit, blocks the (limit+1)th
    - per-IP isolation: one IP tripping doesn't block another
    - window rollover: a fresh window resets the count (monotonic clock
      monkeypatched so we don't sleep 60s)
    - rejected requests don't consume window budget (retry stays blocked
      until rollover, then succeeds)
    - RetryAfter is a positive whole-second hint
    - client_ip: XFF left-most wins; falls back to socket peer; unknown
    - enforce_ip_rate_limit: no-op when DEMO_MODE off; raises when on

The limiter is in-memory only, so tests instantiate fresh IPRateLimiter()
objects rather than the module singleton. The DEMO_MODE gate on
enforce_ip_rate_limit is exercised via monkeypatching config.settings.demo_mode,
mirroring the cost_counter test style.
"""

import pytest

# Reach through the module directly (not via `from llm_guardrails.ratelimit
# import time`) because module-attribute patches
# (`monkeypatch.setattr(rl.time, "monotonic", …)`) only rebind names in the
# module the attribute is looked up on, and `IPRateLimiter.check` looks up
# `time` in this module.
import llm_guardrails.ratelimit as rl
from llm_guardrails.fastapi_ext import client_ip, make_enforce_ip_rate_limit
from llm_guardrails.ratelimit import IPRateLimiter, RateLimitExceeded

# ── helpers ───────────────────────────────────────────────────────────────────


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    """Minimal stand-in for starlette.Request: headers + client only."""

    def __init__(self, headers=None, client_host=None):
        self.headers = headers or {}
        self.client = _FakeClient(client_host) if client_host is not None else None


# ── IPRateLimiter.check ────────────────────────────────────────────────────────


def test_check_allows_up_to_limit():
    lim = IPRateLimiter()
    for _ in range(5):
        lim.check("1.2.3.4", limit=5)  # 5 allowed, no raise


def test_check_blocks_over_limit():
    lim = IPRateLimiter()
    for _ in range(5):
        lim.check("1.2.3.4", limit=5)
    with pytest.raises(RateLimitExceeded) as ei:
        lim.check("1.2.3.4", limit=5)  # 6th in window
    assert ei.value.scope == "ip"
    assert ei.value.retry_after >= 1


def test_check_isolated_per_ip():
    lim = IPRateLimiter()
    for _ in range(5):
        lim.check("1.1.1.1", limit=5)
    # A different IP starts with a clean window.
    lim.check("2.2.2.2", limit=5)
    with pytest.raises(RateLimitExceeded):
        lim.check("1.1.1.1", limit=5)


def test_window_rollover_resets_count(monkeypatch):
    lim = IPRateLimiter()
    clock = {"t": 1000.0}
    monkeypatch.setattr(rl.time, "monotonic", lambda: clock["t"])
    for _ in range(5):
        lim.check("1.2.3.4", limit=5)
    with pytest.raises(RateLimitExceeded):
        lim.check("1.2.3.4", limit=5)
    # Advance past the 60s window — count resets, requests flow again.
    clock["t"] += 61.0
    for _ in range(5):
        lim.check("1.2.3.4", limit=5)


def test_rejected_request_does_not_consume_budget(monkeypatch):
    lim = IPRateLimiter()
    clock = {"t": 0.0}
    monkeypatch.setattr(rl.time, "monotonic", lambda: clock["t"])
    for _ in range(3):
        lim.check("1.2.3.4", limit=3)
    # Several rejected attempts mid-window must not push window_start
    # forward or otherwise extend the lockout.
    for _ in range(4):
        with pytest.raises(RateLimitExceeded):
            lim.check("1.2.3.4", limit=3)
    clock["t"] += 61.0
    lim.check("1.2.3.4", limit=3)  # window rolled — allowed again


def test_reset_clears_windows():
    lim = IPRateLimiter()
    for _ in range(5):
        lim.check("1.2.3.4", limit=5)
    lim.reset()
    for _ in range(5):
        lim.check("1.2.3.4", limit=5)  # clean slate, no raise


# ── client_ip ──────────────────────────────────────────────────────────────────


def test_client_ip_prefers_xff_left_most():
    req = _FakeRequest(
        headers={"x-forwarded-for": "203.0.113.7, 70.0.0.1, 10.0.0.1"},
        client_host="10.0.0.1",
    )
    assert client_ip(req) == "203.0.113.7"


def test_client_ip_falls_back_to_socket_peer():
    req = _FakeRequest(headers={}, client_host="198.51.100.9")
    assert client_ip(req) == "198.51.100.9"


def test_client_ip_unknown_when_no_client():
    req = _FakeRequest(headers={}, client_host=None)
    assert client_ip(req) == "unknown"


def test_client_ip_ignores_blank_xff():
    req = _FakeRequest(headers={"x-forwarded-for": "   "}, client_host="198.51.100.9")
    assert client_ip(req) == "198.51.100.9"


# ── enforce_ip_rate_limit dependency ──────────────────────────────────────────


def test_enforce_noop_when_disabled():
    """Factory-built dep is a no-op when enabled=False, regardless of load."""
    limiter = IPRateLimiter()
    enforce = make_enforce_ip_rate_limit(limiter, cap_rpm=1, enabled=False)
    req = _FakeRequest(client_host="5.5.5.5")
    # Far over the cap, but enabled=False → never raises.
    for _ in range(10):
        enforce(req)


def test_enforce_binds_when_enabled():
    """Factory-built dep enforces the cap when enabled=True."""
    limiter = IPRateLimiter()
    enforce = make_enforce_ip_rate_limit(limiter, cap_rpm=3, enabled=True)
    req = _FakeRequest(client_host="6.6.6.6")
    for _ in range(3):
        enforce(req)
    with pytest.raises(RateLimitExceeded):
        enforce(req)
