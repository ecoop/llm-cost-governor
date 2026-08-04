# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Per-scope USD budget tracking and enforcement.

`ScopeBudget` tracks cumulative spend against a configurable USD ceiling
and supports a pre-flight check that refuses a call when its estimated
cost would push the scope over the limit. The budget itself is
pricing-agnostic: `would_exceed` takes an already-estimated cost in USD,
so the pricing lookup lives in the caller (see `ScopeBudgetHook`).
`BudgetExceeded` is the exception callers raise and catch.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_cost_governor.pricing import usd_for_usage
from llm_cost_governor.schemas import UsageRecord
from llm_cost_governor.wrapper import CallContext


class BudgetExceeded(Exception):
    """Raised pre-flight when the next call would push the scope over budget.

    The message is user-facing: it names the env var and explains how to
    raise the limit or start a fresh session.
    """


class ScopeBudget:
    """Tracks cumulative LLM spend within a scope and enforces a USD ceiling.

    The mechanism is scope-agnostic — it works for a session, a request,
    a CLI run, or any other bounded slice of time the caller wants to cap.

    Usage::

        budget = ScopeBudget(limit_usd=1.00)

        # before each API call:
        if budget.would_exceed(estimated_cost_usd):
            raise BudgetExceeded(...)

        # after each successful call:
        budget.record(usage_dict)

        # at any time:
        print(budget.spent_usd)
    """

    def __init__(self, limit_usd: float) -> None:
        self._limit = limit_usd
        self._spent = 0.0

    def would_exceed(self, estimated_cost: float) -> bool:
        """Return True if this call's estimated cost would push over the limit.

        Args:
            estimated_cost: Pre-flight USD estimate for the next call. The
                caller prices the call (e.g. via `pricing.usd_for_usage`)
                and passes the resulting dollar figure — the budget stays
                out of the pricing table.

        Returns:
            True iff `spent_usd + estimated_cost > limit_usd`. Callers
            should raise `BudgetExceeded` and abort the API call.
        """
        return (self._spent + estimated_cost) > self._limit

    def record(self, usage_dict: dict) -> None:
        """Add actual token counts from a completed call to the running total.

        Args:
            usage_dict: Token-accounting dict. Expected keys: `model`,
                `input_tokens`, `output_tokens`, `cache_read_input_tokens`,
                `cache_creation_input_tokens`. Missing keys default to 0
                or `""` (unknown model → 0 cost, see `pricing.usd_for_usage`).
        """
        self._spent += usd_for_usage(usage_dict)

    @property
    def spent_usd(self) -> float:
        """Cumulative cost recorded so far in this scope, in USD."""
        return self._spent

    @property
    def limit_usd(self) -> float:
        """Configured ceiling for this scope, in USD."""
        return self._limit


def _default_budget_message(budget: ScopeBudget) -> str:
    """Default `BudgetExceeded` message when no template is supplied."""
    return (
        f"Scope budget ${budget.limit_usd:.2f} would be exceeded "
        f"by this call. Spent so far: ${budget.spent_usd:.4f}."
    )


class ScopeBudgetHook:
    """`Hook` adapter that enforces a `ScopeBudget` in a `guarded_call`.

    `pre` prices the call from its `TokenEstimate` (via the public
    `pricing.usd_for_usage`), runs the `would_exceed` check, and raises
    `BudgetExceeded` when the ceiling would be crossed. `post` records the
    actual cost from the returned `UsageRecord` against the running total.

    Register one instance per scope (per session, per request, per CLI
    run): the hook holds a reference to a specific `ScopeBudget` object,
    so the scope lifetime tracks that object's lifetime.

    `message_template` lets callers override the user-facing message on
    the raised `BudgetExceeded` — Pitchcraft, for instance, ships a
    Session-aware message that names the env var and tells the user how
    to reload. The default is a generic one-liner.
    """

    name = "scope_budget"

    def __init__(
        self,
        budget: ScopeBudget,
        *,
        message_template: Callable[[ScopeBudget], str] | None = None,
    ) -> None:
        self.budget = budget
        self._message_template = message_template or _default_budget_message

    def pre(self, ctx: CallContext) -> None:
        estimated_cost = usd_for_usage({
            "model": ctx.model,
            "input_tokens": ctx.estimate.input_tokens,
            "output_tokens": ctx.estimate.output_tokens,
        })
        if self.budget.would_exceed(estimated_cost):
            raise BudgetExceeded(self._message_template(self.budget))

    def post(self, ctx: CallContext, usage: UsageRecord) -> None:
        self.budget.record({
            "model": usage.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        })
