# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Unit tests for RequirePricedModelHook — the pre-flight unpriced-model gate (#10)."""

import pytest

from llm_cost_governor.budget import (
    BudgetExceeded,
    RequirePricedModelHook,
    ScopeBudget,
    ScopeBudgetHook,
    build_budget_chain,
)
from llm_cost_governor.counters import CostCounter
from llm_cost_governor.pricing import UnpricedModel, is_priced
from llm_cost_governor.schemas import UsageRecord
from llm_cost_governor.wrapper import CallContext, TokenEstimate


class InMemoryBackend:
    """StateBackend duck-type: an in-process object store (see test_state_backends)."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def read(self, object_name: str) -> str | None:
        return self._d.get(object_name)

    def write(self, object_name: str, text: str) -> None:
        self._d[object_name] = text


def _ctx(model: str, input_tokens: int = 1_000_000, output_tokens: int = 1_000_000):
    return CallContext(
        provider="anthropic",
        model=model,
        kwargs={},
        tags={},
        estimate=TokenEstimate(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ── is_priced() ────────────────────────────────────────────────────────────────

def test_is_priced_tracks_the_rate_table():
    assert is_priced("claude-opus-5")
    assert not is_priced("gpt-4o")
    assert not is_priced("claude-haiku-4-5-20251001")  # dated ids never resolve


# ── the hook ───────────────────────────────────────────────────────────────────

def test_priced_model_passes():
    RequirePricedModelHook().pre(_ctx("claude-opus-5"))


def test_unpriced_model_is_refused_preflight():
    with pytest.raises(UnpricedModel, match="gpt-4o"):
        RequirePricedModelHook().pre(_ctx("gpt-4o"))


def test_exempt_model_passes_unpriced():
    # Providers metered in units this library can't express yet (#4) opt out
    # explicitly, so the exemption is greppable rather than silent.
    RequirePricedModelHook(exempt={"tavily-search"}).pre(_ctx("tavily-search"))


def test_empty_model_passes():
    # An empty model id means the usage dict never carried one — not an
    # unknown model. `_cost` already treats these separately (no alert).
    RequirePricedModelHook().pre(_ctx(""))


def test_message_template_is_overridable():
    hook = RequirePricedModelHook(message_template=lambda m: f"nope: {m}")
    with pytest.raises(UnpricedModel, match="nope: gpt-4o"):
        hook.pre(_ctx("gpt-4o"))


def test_post_is_a_noop():
    usage = UsageRecord(
        provider="anthropic", model="gpt-4o", input_tokens=1, output_tokens=1,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
        cost_usd=0.0, tags={}, ts=0.0,
    )
    RequirePricedModelHook().post(_ctx("gpt-4o"), usage)  # must not raise


# ── the hole it closes, in both enforcement paths ──────────────────────────────

def test_scope_budget_alone_never_blocks_an_unpriced_model():
    # Documents the fail-open this hook exists to fix: the estimate prices at
    # $0 so the call proceeds, post records $0 so the total never grows, so no
    # later call is blocked either. The model is exempt, not merely cheap.
    budget = ScopeBudget(limit_usd=1.00)
    hook = ScopeBudgetHook(budget)
    for _ in range(50):
        hook.pre(_ctx("gpt-4o"))  # never raises
    assert budget.spent_usd == 0.0


def test_require_priced_blocks_what_scope_budget_lets_through():
    budget = ScopeBudget(limit_usd=1.00)
    hooks = [RequirePricedModelHook(), ScopeBudgetHook(budget)]
    with pytest.raises(UnpricedModel):
        for h in hooks:
            h.pre(_ctx("gpt-4o"))


def test_cost_counter_windows_never_grow_for_an_unpriced_model():
    # The cap path fails for a different reason than the budget path:
    # WindowedCapHook.pre doesn't price the call at all, it compares
    # already-recorded totals. So the fail-open is entirely in record() —
    # real (unpatched) pricing gives $0, the windows stay flat, and enforce()
    # has nothing to trip on no matter how much was actually spent.
    counter = CostCounter(
        object_name="cost_counter.json",
        backend=InMemoryBackend(),
        enabled=True,
        hourly_cap_usd=0.50,
        daily_cap_usd=2.00,
        weekly_cap_usd=10.00,
        per_token_cap_usd=1.00,
    )
    for _ in range(10):
        counter.record(
            {"model": "gpt-4o", "input_tokens": 50_000_000, "output_tokens": 50_000_000},
            token=None,
        )
    counter.enforce(token=None)  # must not raise — the windows never grew
    assert all(v == 0.0 for k, v in counter.current_usage().items()
               if k.endswith("_usd"))


def test_priced_model_still_moves_the_budget():
    # Guard against the gate being so strict it breaks the happy path.
    budget = ScopeBudget(limit_usd=100.00)
    hooks = [RequirePricedModelHook(), ScopeBudgetHook(budget)]
    ctx = _ctx("claude-opus-5")
    for h in hooks:
        h.pre(ctx)
    usage = UsageRecord(
        provider="anthropic", model="claude-opus-5",
        input_tokens=1_000_000, output_tokens=1_000_000,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
        cost_usd=30.0, tags={}, ts=0.0,
    )
    hooks[1].post(ctx, usage)
    assert budget.spent_usd == pytest.approx(30.00)


# ── build_budget_chain (#10 follow-up: the half-configured footgun) ────────────

def test_chain_from_a_limit():
    hooks = build_budget_chain(3.00)
    assert [h.name for h in hooks] == ["require_priced_model", "scope_budget"]
    assert hooks[1].budget.limit_usd == pytest.approx(3.00)


def test_chain_from_an_existing_budget_shares_the_instance():
    # Passing your own budget is how you read spent_usd afterwards, or share
    # one ceiling across several call sites in a run.
    b = ScopeBudget(limit_usd=5.00)
    hooks = build_budget_chain(budget=b)
    assert hooks[1].budget is b


def test_chain_requires_exactly_one_of_limit_or_budget():
    # Silently picking one would reintroduce the half-configured state this
    # function exists to prevent.
    with pytest.raises(ValueError, match="exactly one"):
        build_budget_chain()
    with pytest.raises(ValueError, match="exactly one"):
        build_budget_chain(1.00, budget=ScopeBudget(limit_usd=2.00))


def test_chain_forwards_exempt_to_the_gate():
    hooks = build_budget_chain(3.00, exempt={"tavily-search"})
    hooks[0].pre(_ctx("tavily-search"))  # admitted, not refused


def test_chain_forwards_message_template_to_the_budget():
    hooks = build_budget_chain(0.01, message_template=lambda b: "custom cap msg")
    with pytest.raises(BudgetExceeded, match="custom cap msg"):
        for h in hooks:
            h.pre(_ctx("claude-opus-5"))


def test_chain_actually_gates_an_unpriced_model():
    hooks = build_budget_chain(1000.00)
    with pytest.raises(UnpricedModel):
        for h in hooks:
            h.pre(_ctx("gpt-4o"))


def test_chain_admits_a_priced_model_within_budget():
    hooks = build_budget_chain(1000.00)
    for h in hooks:
        h.pre(_ctx("claude-opus-5"))
