# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Tests for the guardrails wrapper — HookChain, guarded_call, record_usage.

The wrapper is provider-agnostic in principle; here it's exercised
against a hand-rolled fake `Anthropic`-shaped client so the tests don't
need the SDK or a network. Behaviors covered:

  * `HookChain.run_pre` — hooks fire in order; a raise propagates.
  * `HookChain.run_post` — hooks fire in order; a raise is caught and
    logged, later hooks still run.
  * `guarded_call` — happy path composes correctly, missing model
    raises, an injection hook bypasses the SDK, truncation raises
    after post-hooks (so accounting still happens for the wasted call).
  * `record_usage` — post-hooks fire, no SDK call, UsageRecord priced
    from the caller-supplied token counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_guardrails.schemas import TokenEstimate, UsageRecord
from llm_guardrails.wrapper import (
    CallContext,
    HookChain,
    TruncationError,
    guarded_call,
    record_usage,
)


# ── Test doubles ──────────────────────────────────────────────────────────────


@dataclass
class _FakeAnthropicUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


@dataclass
class _FakeAnthropicResponse:
    usage: _FakeAnthropicUsage
    stop_reason: str = "end_turn"


class _FakeMessages:
    def __init__(self, response: _FakeAnthropicResponse):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
        self.calls.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeAnthropicResponse | None = None):
        self.messages = _FakeMessages(response or _FakeAnthropicResponse(_FakeAnthropicUsage()))


@dataclass
class _RecordingHook:
    """Records pre/post invocations for assertion by tests."""

    name: str
    pre_raises: Exception | None = None
    post_raises: Exception | None = None
    stashes_injection: Any = None
    pre_calls: list[CallContext] = field(default_factory=list)
    post_calls: list[tuple[CallContext, UsageRecord]] = field(default_factory=list)

    def pre(self, ctx: CallContext) -> None:
        self.pre_calls.append(ctx)
        if self.stashes_injection is not None:
            ctx.state["_injected_response"] = self.stashes_injection
        if self.pre_raises is not None:
            raise self.pre_raises

    def post(self, ctx: CallContext, usage: UsageRecord) -> None:
        self.post_calls.append((ctx, usage))
        if self.post_raises is not None:
            raise self.post_raises


# ── HookChain ─────────────────────────────────────────────────────────────────


def _ctx(model: str = "claude-sonnet-4-6") -> CallContext:
    return CallContext(
        provider="anthropic",
        model=model,
        kwargs={"model": model},
        tags={},
        estimate=TokenEstimate(input_tokens=0, output_tokens=0),
    )


def _usage(model: str = "claude-sonnet-4-6") -> UsageRecord:
    return UsageRecord(
        provider="anthropic", model=model,
        input_tokens=0, output_tokens=0, cost_usd=0.0,
    )


def test_run_pre_calls_hooks_in_order():
    order: list[str] = []

    class _H:
        name = "h"
        def __init__(self, tag): self.tag = tag
        def pre(self, ctx): order.append(self.tag)
        def post(self, ctx, usage): pass

    HookChain([_H("a"), _H("b"), _H("c")]).run_pre(_ctx())
    assert order == ["a", "b", "c"]


def test_run_pre_raise_propagates_and_skips_later_hooks():
    h_a = _RecordingHook("a")
    h_b = _RecordingHook("b", pre_raises=RuntimeError("stop"))
    h_c = _RecordingHook("c")

    with pytest.raises(RuntimeError, match="stop"):
        HookChain([h_a, h_b, h_c]).run_pre(_ctx())

    assert len(h_a.pre_calls) == 1
    assert len(h_b.pre_calls) == 1
    assert h_c.pre_calls == []  # never reached


def test_run_post_calls_hooks_in_order():
    order: list[str] = []

    class _H:
        name = "h"
        def __init__(self, tag): self.tag = tag
        def pre(self, ctx): pass
        def post(self, ctx, usage): order.append(self.tag)

    HookChain([_H("a"), _H("b"), _H("c")]).run_post(_ctx(), _usage())
    assert order == ["a", "b", "c"]


def test_run_post_raise_is_swallowed_and_later_hooks_run(caplog):
    h_a = _RecordingHook("a")
    h_b = _RecordingHook("b", post_raises=RuntimeError("boom"))
    h_c = _RecordingHook("c")

    HookChain([h_a, h_b, h_c]).run_post(_ctx(), _usage())

    assert len(h_a.post_calls) == 1
    assert len(h_b.post_calls) == 1
    assert len(h_c.post_calls) == 1  # continued past the raise
    assert any("hook b post failed" in rec.message for rec in caplog.records)


# ── guarded_call: happy path + validation ─────────────────────────────────────


def test_guarded_call_returns_response_and_priced_usage():
    client = _FakeClient(_FakeAnthropicResponse(
        _FakeAnthropicUsage(input_tokens=1_000_000, output_tokens=0),
    ))

    response, usage = guarded_call(
        client, provider="anthropic", hooks=(),
        tags={"stage": "drafter"},
        model="claude-sonnet-4-6", messages=[], max_tokens=1024,
    )

    assert response is client.messages._response
    assert client.messages.calls == [{
        "model": "claude-sonnet-4-6", "messages": [], "max_tokens": 1024,
    }]
    # Sonnet 4.6 at $3/M input, 1M input → $3.00 cost.
    assert usage.cost_usd == pytest.approx(3.00)
    assert usage.model == "claude-sonnet-4-6"
    assert usage.tags == {"stage": "drafter"}
    assert usage.provider == "anthropic"


def test_guarded_call_requires_model_in_kwargs():
    client = _FakeClient()
    with pytest.raises(ValueError, match="`model` is required"):
        guarded_call(client, provider="anthropic", messages=[], max_tokens=100)


def test_guarded_call_unknown_provider_raises():
    client = _FakeClient()
    with pytest.raises(ValueError, match="Unknown provider"):
        guarded_call(client, provider="chatgpt-4-9000", model="foo",
                     messages=[], max_tokens=10)


# ── guarded_call: hook chain composition ──────────────────────────────────────


def test_guarded_call_runs_pre_then_provider_then_post():
    client = _FakeClient()
    h = _RecordingHook("logger")

    _response, usage = guarded_call(
        client, provider="anthropic", hooks=[h],
        model="claude-sonnet-4-6", messages=[], max_tokens=1024,
    )

    # pre saw the ctx before the call; post saw the usage after.
    assert len(h.pre_calls) == 1
    assert h.pre_calls[0].model == "claude-sonnet-4-6"
    assert len(h.post_calls) == 1
    assert h.post_calls[0][1] is usage


def test_guarded_call_pre_hook_raise_aborts_before_provider_call():
    client = _FakeClient()
    h = _RecordingHook("gate", pre_raises=RuntimeError("nope"))

    with pytest.raises(RuntimeError, match="nope"):
        guarded_call(
            client, provider="anthropic", hooks=[h],
            model="claude-sonnet-4-6", messages=[], max_tokens=10,
        )

    # SDK was never invoked
    assert client.messages.calls == []
    # post-hook never fires when pre aborts
    assert h.post_calls == []


def test_guarded_call_injection_bypasses_provider():
    client = _FakeClient()
    fake_response = _FakeAnthropicResponse(
        _FakeAnthropicUsage(input_tokens=42, output_tokens=7),
    )
    injector = _RecordingHook("injector", stashes_injection=fake_response)
    logger = _RecordingHook("logger")

    response, usage = guarded_call(
        client, provider="anthropic", hooks=[injector, logger],
        model="claude-sonnet-4-6", messages=[], max_tokens=10,
    )

    # Provider SDK never called; response is the injected one.
    assert client.messages.calls == []
    assert response is fake_response
    # Post-hook still ran with the priced usage.
    assert len(logger.post_calls) == 1
    assert usage.input_tokens == 42
    assert usage.output_tokens == 7


# ── guarded_call: truncation ──────────────────────────────────────────────────


def test_guarded_call_truncation_raises_after_post_hooks():
    client = _FakeClient(_FakeAnthropicResponse(
        _FakeAnthropicUsage(input_tokens=1_000_000, output_tokens=0),
        stop_reason="max_tokens",
    ))
    logger = _RecordingHook("logger")

    with pytest.raises(TruncationError, match="max_tokens=1024"):
        guarded_call(
            client, provider="anthropic", hooks=[logger],
            model="claude-sonnet-4-6", messages=[], max_tokens=1024,
        )

    # Post-hook DID fire — the call cost real money and must be accounted for.
    assert len(logger.post_calls) == 1
    assert logger.post_calls[0][1].cost_usd == pytest.approx(3.00)


# ── record_usage ──────────────────────────────────────────────────────────────


def test_record_usage_prices_and_runs_post_hooks_only():
    logger = _RecordingHook("logger")

    usage = record_usage(
        provider="anthropic", model="claude-sonnet-4-6",
        input_tokens=1_000_000, output_tokens=0,
        hooks=[logger], tags={"call_type": "embedding", "qa_id": "q42"},
    )

    # Priced correctly.
    assert usage.cost_usd == pytest.approx(3.00)
    assert usage.tags == {"call_type": "embedding", "qa_id": "q42"}
    # Post-hook saw it; pre-hook path skipped (no SDK call).
    assert logger.pre_calls == []
    assert len(logger.post_calls) == 1
    assert logger.post_calls[0][1] is usage


def test_record_usage_preserves_cache_field_semantics():
    # `None` (provider doesn't expose) vs `0` (provider said zero) must
    # survive the trip through record_usage — accounting downstream cares.
    priced = record_usage(
        provider="voyage", model="unknown-model-abc",
        input_tokens=1000, output_tokens=0,
        cache_read_input_tokens=None, cache_creation_input_tokens=0,
    )
    assert priced.cache_read_input_tokens is None
    assert priced.cache_creation_input_tokens == 0
    # Unknown model → cost is 0.0, no crash.
    assert priced.cost_usd == 0.0
