# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Per-scope USD budget tracking and enforcement.

`ScopeBudget` tracks cumulative spend against a configurable USD ceiling
and supports a pre-flight check that refuses a call when its estimated
cost would push the scope over the limit. The budget itself is
pricing-agnostic: `would_exceed` takes an already-estimated cost in USD,
so the pricing lookup lives in the caller (see `ScopeBudgetHook`).
`BudgetExceeded` is the exception callers raise and catch.

`RequirePricedModelHook` is the separate guard that closes the fail-open
hole underneath every spend control: a model with no rate row costs $0,
so it never moves any ledger and no cap can ever trip for it. The hook
refuses such a call pre-flight, while aborting is still free.

`build_budget_chain` assembles the two into a hard, gated ceiling in one
call. Prefer it over hand-assembling the pair: it is a single import
that either resolves or raises `ImportError` loudly, which is what stops
a too-old install from silently degrading to no enforcement at all.
"""

from __future__ import annotations

from collections.abc import Callable, Collection

from llm_cost_governor.pricing import UnpricedModel, is_priced, usd_for_usage
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


def _default_unpriced_message(model: str) -> str:
    """Default message for a refused unpriced call."""
    return (
        f"Model {model!r} has no entry in pricing.MODEL_PRICING, so its spend "
        f"would be counted as $0 and would not move any budget or cap. "
        f"Add a rate row, or pass the id in RequirePricedModelHook(exempt=...) "
        f"if it is metered elsewhere."
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


class RequirePricedModelHook:
    """`Hook` that refuses a call whose model has no rate row.

    Spend controls in this library are all downstream of one number: the
    USD a call is priced at. `pricing._cost` returns ``0.0`` for a model
    it has no row for — deliberately, so that pricing-table lag can never
    crash a live session. The cost of that tolerance is that an unpriced
    model is not merely billed loosely, it is **exempt**: it adds $0 to
    every budget and every rolling window, so those totals never grow and
    no ceiling downstream of them can ever trip.

    Both enforcement paths fail this way, for different reasons:

    * `ScopeBudgetHook.pre` prices the call's estimate, gets $0, and
      lets it through; its `post` then records $0, so the running total
      never rises and no later call is blocked either.
    * `WindowedCapHook.pre` doesn't price the call at all — it compares
      *already-recorded* window totals. Its `post` records $0, so the
      windows never grow and the caps never fire.

    A post-flight fix cannot help: by then the money is spent, and
    `HookChain.run_post` swallows hook exceptions by contract so nothing
    raised there would even reach the caller. Pre-flight is the only
    place where refusing costs nothing but an aborted request — which is
    what this hook does.

    Register it *before* the spend hooks so it runs first::

        hooks = [
            RequirePricedModelHook(),
            ScopeBudgetHook(budget),
        ]

    It is deliberately opt-in rather than folded into the spend hooks:
    apps that price some traffic outside this library, or that meter
    providers with no token dimension at all, have a legitimate reason
    to let an unpriced model through. Those ids go in `exempt`, so the
    exemption is explicit and greppable instead of silent.

    Args:
        exempt: Model ids allowed through unpriced. Use for providers
            billed in units this library cannot yet express — a
            per-request search API, say — where $0 from the token math
            is expected rather than a symptom. Anything here is
            unmetered: exempt an id only when its spend is bounded some
            other way.
        message_template: Overrides the exception message. Receives the
            offending model id.
    """

    name = "require_priced_model"

    def __init__(
        self,
        exempt: Collection[str] = (),
        message_template: Callable[[str], str] | None = None,
    ) -> None:
        self.exempt = frozenset(exempt)
        self._message_template = message_template or _default_unpriced_message

    def pre(self, ctx: CallContext) -> None:
        """Refuse the call when `ctx.model` has no rate row.

        Raises:
            UnpricedModel: When the model is neither priced nor exempt.
        """
        if not ctx.model or ctx.model in self.exempt:
            return
        if not is_priced(ctx.model):
            raise UnpricedModel(self._message_template(ctx.model))

    def post(self, ctx: CallContext, usage: UsageRecord) -> None:
        """No-op. The decision this hook makes is only meaningful pre-flight."""


def build_budget_chain(
    limit_usd: float | None = None,
    *,
    budget: ScopeBudget | None = None,
    exempt: Collection[str] = (),
    message_template: Callable[[ScopeBudget], str] | None = None,
) -> list[object]:
    """Assemble a hard, gated USD ceiling as one hook chain.

    Returns ``[RequirePricedModelHook(...), ScopeBudgetHook(...)]`` — the
    unpriced-model gate followed by the budget. Hand it straight to
    ``guarded_call(hooks=...)``.

    **Why prefer this over assembling the pair by hand.** A caller that
    guards the import for compatibility with older releases::

        try:
            from llm_cost_governor.budget import (
                RequirePricedModelHook, ScopeBudget, ScopeBudgetHook,
            )
        except ImportError:
            return []                      # ← drops the CEILING too

    silently loses *all* enforcement on an install predating the gate,
    not just the gate. The failure is invisible: no exception, no log,
    an app that believes it has a ceiling and does not. Importing this
    one symbol instead either resolves or raises `ImportError` at the
    import site, where it is obvious and actionable.

    Args:
        limit_usd: Ceiling in USD; a `ScopeBudget` is constructed for it.
            Give exactly one of this or `budget`.
        budget: An existing `ScopeBudget` to enforce. Pass this when you
            need to read `spent_usd` afterwards, or to share one ceiling
            across several call sites in the same run.
        exempt: Model ids the gate should admit unpriced — forwarded to
            `RequirePricedModelHook`. For providers metered in units this
            library cannot express. Anything listed is unmetered.
        message_template: Overrides the `BudgetExceeded` message;
            forwarded to `ScopeBudgetHook`.

    Returns:
        The hooks, gate first. Order is not load-bearing — `run_pre`
        executes every pre-hook and any raise aborts before the SDK call
        — but gate-first surfaces the more specific exception when a
        model is both unpriced and over budget.

    Raises:
        ValueError: If neither or both of `limit_usd` and `budget` are
            given. Silently picking one would reintroduce exactly the
            half-configured state this function exists to prevent.

    Example:
        >>> hooks = build_budget_chain(3.00)
        >>> # or, to read spend afterwards:
        >>> b = ScopeBudget(limit_usd=3.00)
        >>> hooks = build_budget_chain(budget=b)
    """
    if (limit_usd is None) == (budget is None):
        raise ValueError(
            "build_budget_chain: give exactly one of `limit_usd` or `budget` "
            f"(got limit_usd={limit_usd!r}, budget={budget!r})"
        )
    scope = budget if budget is not None else ScopeBudget(limit_usd=float(limit_usd))
    return [
        RequirePricedModelHook(exempt=exempt),
        ScopeBudgetHook(scope, message_template=message_template),
    ]
