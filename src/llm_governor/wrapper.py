# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Hook chain + `guarded_call` — the LLM-call composition point.

`guarded_call` is the library's single entry point for wrapping an
LLM SDK call with a caller-registered sequence of enforcement +
accounting hooks. Its shape:

    response, usage = guarded_call(
        client, provider="anthropic", hooks=[...],
        tags={"stage": "drafter"},
        model="claude-opus-4-8", messages=[...], max_tokens=8192,
    )

Hook contract (see `Hook` Protocol):

  * `pre(ctx)` runs before the SDK call. May raise to abort the call
    (budget exceeded, cap exceeded, ratelimit exceeded). Pre-hooks
    should be side-effect-free reads or reservations stashed in
    `ctx.state` for a paired post-hook to clear — the chain does not
    roll back earlier pre-hooks when a later one aborts.

  * `post(ctx, usage)` runs after the SDK call. Records spend, logs
    the event, closes the span, etc. Post-hooks are best-effort: a
    failure in one is logged and swallowed, remaining hooks still run,
    and the caller still receives its response.

Injection hook pattern: a pre-hook may stash a canned response in
`ctx.state["_injected_response"]`; the wrapper returns that instead of
calling the provider. Used by Pitchcraft's FAKE_LLM mode to replay
recorded responses without touching the API.

Provider adapter (see `llm_governor/providers/`): supplies the
`call` / `extract_usage` / `truncated` / `default_estimate` seams so
the wrapper stays provider-agnostic.

`record_usage` is the sibling for calls the app made itself (e.g.
Voyage embeddings, batch APIs, vision-extract CLI). It runs only the
post-hooks — no pre-flight, no wrapped call — and lets those calls
share the same accounting/logging path as `guarded_call`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llm_governor.pricing import usd_for_usage
from llm_governor.providers import get_provider
from llm_governor.schemas import TokenEstimate, UsageRecord

_log = logging.getLogger(__name__)


class TruncationError(Exception):
    """Raised when the provider truncated the response at ``max_tokens``.

    Raised *after* the call has completed and all post-hooks have run —
    the call cost real money and must be accounted for even when the
    response is unusable. Callers that want to retry with a larger
    budget catch this exception.
    """


# ── Call context passed through the hook chain ────────────────────────────────


@dataclass
class CallContext:
    """State passed to every hook's `pre` and `post`.

    Attributes:
        provider: The provider name (`"anthropic"`, `"openai"`, ...).
        model: The model id from the outbound kwargs.
        kwargs: The full kwargs about to be forwarded to the provider
            SDK's create call. Hooks may inspect but should not mutate
            unless they know what they're doing (e.g. injection).
        tags: The caller's opaque annotation space, forwarded to
            ``UsageRecord.tags`` and available to hooks for context-
            aware behavior.
        estimate: The pre-flight token estimate (provider default,
            or a caller override).
        state: Scratch space for hook-to-hook handoff — e.g. an
            OTelSpanHook opens a span in `pre` and closes it in `post`
            by looking it up here; a FakeLLMHook stashes an injected
            response for the wrapper to return.
    """

    provider: str
    model: str
    kwargs: dict
    tags: dict[str, str]
    estimate: TokenEstimate
    state: dict[str, Any] = field(default_factory=dict)


# ── Hook Protocol + chain runner ──────────────────────────────────────────────


@runtime_checkable
class Hook(Protocol):
    """One step in the guarded_call pre/post chain.

    Concrete hook classes should set a ``name`` attribute (used for
    logging when a post-hook raises); implementations must define
    both ``pre`` and ``post`` even if one is a no-op.
    """

    name: str

    def pre(self, ctx: CallContext) -> None:
        """Run before the SDK call. May raise to abort."""
        ...

    def post(self, ctx: CallContext, usage: UsageRecord) -> None:
        """Run after the SDK call. Failures are logged and swallowed."""
        ...


class HookChain:
    """Runs the registered pre and post hooks in order.

    Pre-hooks propagate any exception (aborting the call). Post-hooks
    are best-effort — a failure in one is logged and the loop
    continues, since accounting shouldn't be able to break a request
    that already succeeded.
    """

    def __init__(self, hooks: Sequence[Hook]) -> None:
        self.hooks = list(hooks)

    def run_pre(self, ctx: CallContext) -> None:
        """Run every pre-hook; any raise aborts the call and propagates.

        The chain does not roll back earlier pre-hooks when a later one
        aborts. Pre-hooks that acquire a resource / hold a reservation
        must clear it themselves (typically via a matching post-hook,
        or a try/except with a compensating action).
        """
        for h in self.hooks:
            h.pre(ctx)

    def run_post(self, ctx: CallContext, usage: UsageRecord) -> None:
        """Run every post-hook; failures are logged and swallowed.

        A hook that raises does not prevent later hooks from running,
        and does not propagate to the caller. This is deliberate:
        accounting or logging failures must not break a response that
        succeeded.
        """
        for h in self.hooks:
            try:
                h.post(ctx, usage)
            except Exception:  # noqa: BLE001 — post-hooks are best-effort by contract
                _log.warning(
                    "hook %s post failed",
                    getattr(h, "name", type(h).__name__),
                    exc_info=True,
                )


# ── The wrapper ───────────────────────────────────────────────────────────────


def guarded_call(
    client: Any,
    *,
    provider: str = "anthropic",
    hooks: Sequence[Hook] = (),
    tags: Mapping[str, str] | None = None,
    estimate: TokenEstimate | None = None,
    raise_on_truncation: bool = True,
    **create_kwargs: Any,
) -> tuple[Any, UsageRecord]:
    """Invoke the provider's create call, wrapped in the hook chain.

    Args:
        client: The provider SDK client (e.g. ``anthropic.Anthropic``).
        provider: The provider adapter to use — must match a key in
            ``llm_governor.providers.get_provider``.
        hooks: Ordered sequence of hooks to run. Pre-hooks execute in
            registration order; a raise aborts the call. Post-hooks
            execute in registration order; failures are logged and
            swallowed.
        tags: Opaque caller annotations (stage/agent/route/qa_id/…)
            forwarded to ``UsageRecord.tags`` and visible to hooks
            through ``CallContext.tags``.
        estimate: Pre-flight token estimate. If omitted, the provider
            adapter's ``default_estimate`` computes one from
            ``create_kwargs``.
        raise_on_truncation: When True (default) the wrapper raises
            ``TruncationError`` after post-hooks if the provider hit
            ``max_tokens``. Callers that want to inspect / capture a
            truncated response before raising (e.g. Pitchcraft's
            fake-LLM fixture capture path) set this to False and check
            the returned response themselves.
        **create_kwargs: Forwarded verbatim to the provider SDK
            (``client.messages.create`` for Anthropic). Must include
            ``model``.

    Returns:
        (response, UsageRecord) — the raw provider response and the
        normalized usage/cost record.

    Raises:
        ValueError: When ``model`` is missing from ``create_kwargs``.
        TruncationError: When the response hit ``max_tokens`` *and*
            ``raise_on_truncation`` is True. Raised *after* post-hooks
            have run (the call cost real money and must be accounted for).
        Any pre-hook exception: Propagated without a call being made.
    """
    adapter = get_provider(provider)

    model = create_kwargs.get("model")
    if not model:
        raise ValueError("guarded_call: `model` is required in create_kwargs")

    ctx = CallContext(
        provider=provider,
        model=model,
        kwargs=dict(create_kwargs),
        tags=dict(tags or {}),
        estimate=estimate or adapter.default_estimate(create_kwargs),
    )
    chain = HookChain(hooks)

    # Pre-flight: budget checks, cap enforcement, injection, span open, …
    chain.run_pre(ctx)

    # Actual provider call. An injection hook may have stashed a canned
    # response in ctx.state; when present, the wrapper returns that and
    # skips the SDK entirely.
    response = ctx.state.get("_injected_response") or adapter.call(client, ctx.kwargs)

    # Normalize usage and price the call. `usd_for_usage` handles unknown
    # models via the pricing module's warn-once seam ($0 cost + alert).
    tokens = adapter.extract_usage(response)
    cost_usd = usd_for_usage({**tokens, "model": model})

    usage = UsageRecord(
        provider=provider,
        model=model,
        input_tokens=tokens.get("input_tokens", 0),
        output_tokens=tokens.get("output_tokens", 0),
        cache_read_input_tokens=tokens.get("cache_read_input_tokens"),
        cache_creation_input_tokens=tokens.get("cache_creation_input_tokens"),
        cost_usd=cost_usd,
        tags=dict(ctx.tags),
        ts=time.time(),
    )

    # Post-flight: record spend, log event, close span, …
    chain.run_post(ctx, usage)

    # Truncation is a caller concern; raise after accounting so the
    # cost of the wasted call is still recorded. Callers that want to
    # capture the truncated response (fixture recording, retry-with-
    # more-tokens) opt out via `raise_on_truncation=False` and check
    # `adapter.truncated(response)` themselves.
    if raise_on_truncation and adapter.truncated(response):
        raise TruncationError(
            f"{model} response truncated at max_tokens={create_kwargs.get('max_tokens')}"
        )

    return response, usage


def record_usage(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    hooks: Sequence[Hook] = (),
    tags: Mapping[str, str] | None = None,
) -> UsageRecord:
    """Account for a call the app made itself — embeddings, vision, batch.

    Runs the given hooks' ``post`` step only — there's no pre-flight
    and no wrapped call, because the caller has already made the API
    request. Use this for provider calls that don't fit the
    ``guarded_call`` shape (Voyage embeddings, one-shot CLI scripts,
    batch-API returns) but should still be priced, capped, and logged
    through the same seam.

    Args:
        provider: The provider name (used only for the ``UsageRecord``).
        model: The model id — used to look up cost via the pricing table.
        input_tokens / output_tokens: Token counts from the response.
        cache_read_input_tokens / cache_creation_input_tokens: Optional.
            Set to ``None`` when the provider doesn't expose them; ``0``
            when the provider reported zero.
        hooks: Post-hooks to run. Same protocol as ``guarded_call`` —
            failures are logged and swallowed.
        tags: Opaque caller annotations, forwarded to ``UsageRecord.tags``.

    Returns:
        The constructed and hook-processed ``UsageRecord``.
    """
    tokens = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
    }
    cost_usd = usd_for_usage({**tokens, "model": model})

    usage = UsageRecord(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cost_usd=cost_usd,
        tags=dict(tags or {}),
        ts=time.time(),
    )

    ctx = CallContext(
        provider=provider,
        model=model,
        kwargs={},
        tags=dict(tags or {}),
        estimate=TokenEstimate(input_tokens=0, output_tokens=0),
    )
    HookChain(hooks).run_post(ctx, usage)
    return usage
