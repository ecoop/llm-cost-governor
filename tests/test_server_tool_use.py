# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Unit tests for server-side tool metering (#11).

Anthropic bills some server-side tool use on top of tokens and reports it as
a sibling of the token counts. The adapter used to drop that field, so the
spend was invisible — on a fully priced model, with no warning, because
nothing ever looked.
"""

import pytest

from llm_cost_governor.pricing import (
    SERVER_TOOL_PRICING,
    server_tool_cost,
    usd_for_usage,
)
from llm_cost_governor.providers import anthropic as adapter
from llm_cost_governor.wrapper import record_usage


class _Usage:
    def __init__(self, **kw):
        self.input_tokens = kw.pop("input_tokens", 0)
        self.output_tokens = kw.pop("output_tokens", 0)
        for k, v in kw.items():
            setattr(self, k, v)


class _Response:
    def __init__(self, usage):
        self.usage = usage


# ── server_tool_cost() ─────────────────────────────────────────────────────────

def test_web_search_is_priced_per_request():
    # $10 per 1,000 searches.
    assert server_tool_cost({"web_search_requests": 1000}) == pytest.approx(10.00)
    assert server_tool_cost({"web_search_requests": 1}) == pytest.approx(0.010)


def test_web_fetch_is_explicitly_free():
    # Known to be free (tokens only), not forgotten — so it must not warn.
    assert "web_fetch_requests" in SERVER_TOOL_PRICING
    assert server_tool_cost({"web_fetch_requests": 500}) == 0.0


def test_none_and_empty_cost_nothing():
    assert server_tool_cost(None) == 0.0
    assert server_tool_cost({}) == 0.0


def test_unpriced_tool_costs_zero_but_warns_once(monkeypatch):
    # Code execution is billed by container-hour, so it has no per-request
    # rate. It must not silently contribute $0 — that is the exact failure
    # this seam exists to surface.
    from llm_cost_governor import pricing

    monkeypatch.setattr(pricing, "_unpriced_tools_warned", set())
    seen = []
    monkeypatch.setattr(pricing, "alert", lambda *a, **k: seen.append(a), raising=False)
    import llm_cost_governor.alerts as alerts_mod
    monkeypatch.setattr(alerts_mod, "alert", lambda *a, **k: seen.append(a))

    assert pricing.server_tool_cost({"code_execution_requests": 3}) == 0.0
    assert pricing.server_tool_cost({"code_execution_requests": 3}) == 0.0
    assert len(seen) == 1, "should alert once per process, not per call"


# ── it reaches cost_usd, and therefore budgets and caps ────────────────────────

def test_usd_for_usage_includes_server_tool_cost():
    usage = {
        "model": "claude-opus-5",
        "input_tokens": 1_000_000,
        "output_tokens": 0,
        "server_tool_use": {"web_search_requests": 100},
    }
    # $5.00 of tokens + 100 searches at $0.01 = $6.00
    assert usd_for_usage(usage) == pytest.approx(6.00)


def test_usd_for_usage_unchanged_without_server_tools():
    usage = {"model": "claude-opus-5", "input_tokens": 1_000_000, "output_tokens": 0}
    assert usd_for_usage(usage) == pytest.approx(5.00)


def test_record_usage_carries_and_prices_it():
    r = record_usage(
        provider="anthropic", model="claude-opus-5",
        input_tokens=1_000_000, output_tokens=0,
        server_tool_use={"web_search_requests": 100},
    )
    assert r.server_tool_use == {"web_search_requests": 100}
    assert r.cost_usd == pytest.approx(6.00)


def test_record_usage_defaults_to_none():
    r = record_usage(provider="anthropic", model="claude-opus-5",
                     input_tokens=1_000, output_tokens=0)
    assert r.server_tool_use is None


# ── the adapter no longer drops the field ──────────────────────────────────────

def test_adapter_extracts_server_tool_use():
    resp = _Response(_Usage(input_tokens=105, output_tokens=6039,
                            server_tool_use={"web_search_requests": 2}))
    assert adapter.extract_usage(resp)["server_tool_use"] == {"web_search_requests": 2}


def test_adapter_normalizes_an_sdk_object():
    class _STU:
        web_search_requests = 3

    resp = _Response(_Usage(input_tokens=1, output_tokens=1, server_tool_use=_STU()))
    assert adapter.extract_usage(resp)["server_tool_use"] == {"web_search_requests": 3}


def test_adapter_returns_none_when_absent():
    # Matches the cache fields' convention: None means "provider didn't report".
    resp = _Response(_Usage(input_tokens=1, output_tokens=1))
    assert adapter.extract_usage(resp)["server_tool_use"] is None
