# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""`Hook`-Protocol adapters that add OTel/LangSmith spans to `guarded_call`.

Two independent hooks, both optional:

    * ``OTelSpanHook`` — opens one span per ``guarded_call`` in
      ``pre`` and closes it in ``post`` with the priced UsageRecord's
      token counts and cost stamped as attributes. Hands the span off
      pre → post via ``CallContext.state["otel_span"]``.

    * ``LangSmithMetadataHook`` — stamps a caller-supplied identity
      dict as ``langsmith.metadata.*`` attributes on the span the
      SpanHook opened. Independent of the SpanHook (composes with any
      other span-opening hook that stashes an ``"otel_span"`` in
      ``ctx.state``), but requires SOME upstream hook to have opened
      the span. Split from ``OTelSpanHook`` so a consumer that wants
      basic tracing without LangSmith conventions can skip it.

Both hooks are safe if there's no active OTel provider — the SDK's
``trace.get_tracer`` returns a no-op tracer that swallows every call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from opentelemetry import trace

from llm_governor.schemas import UsageRecord
from llm_governor.wrapper import CallContext


class OTelSpanHook:
    """`Hook` adapter that opens one OTel span per `guarded_call`.

    `pre` opens the span, stashes it in ``ctx.state["otel_span"]``, and
    sets the standard GenAI request attributes (provider, model) plus
    each entry in ``ctx.tags`` as ``llm_governor.tag.<key>``.

    `post` sets ``gen_ai.usage.*`` attributes for input/output/cache
    tokens plus ``llm_governor.cost_usd`` from the priced UsageRecord,
    then ends the span. If the span is missing (e.g. some earlier
    exception unwound state), `post` is a no-op — never raises into
    the caller.
    """

    name = "otel_span"

    def __init__(self, tracer_name: str = "llm_governor") -> None:
        self._tracer = trace.get_tracer(tracer_name)

    def pre(self, ctx: CallContext) -> None:
        span = self._tracer.start_span(
            f"llm.{ctx.provider}.call",
            attributes={
                "gen_ai.system": ctx.provider,
                "gen_ai.request.model": ctx.model,
                **{f"llm_governor.tag.{k}": v for k, v in ctx.tags.items()},
            },
        )
        ctx.state["otel_span"] = span

    def post(self, ctx: CallContext, usage: UsageRecord) -> None:
        span = ctx.state.get("otel_span")
        if span is None:
            return
        span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        if usage.cache_read_input_tokens is not None:
            span.set_attribute(
                "gen_ai.usage.cache_read_input_tokens",
                usage.cache_read_input_tokens,
            )
        if usage.cache_creation_input_tokens is not None:
            span.set_attribute(
                "gen_ai.usage.cache_creation_input_tokens",
                usage.cache_creation_input_tokens,
            )
        span.set_attribute("llm_governor.cost_usd", usage.cost_usd)
        span.end()


class LangSmithMetadataHook:
    """`Hook` adapter that stamps LangSmith-metadata attrs on the current call span.

    Depends on an upstream hook (e.g. ``OTelSpanHook``) having stashed
    a span in ``ctx.state["otel_span"]``. Both `pre` and `post` write
    metadata: `pre` for attributes known before the call
    (session/user/persona), `post` for anything the caller only knows
    after the response (rare — most consumers just use `pre`).

    Metadata comes from a caller-supplied ``identity_provider``
    callable so the hook has no direct dependency on Pitchcraft's
    session/identity modules. Ulty-goalty can pass a callable that
    returns ``{route, qa_id}``; Pitchcraft can pass one that returns
    ``{session_id, user_id, persona, invite_token}``. The library
    stays app-neutral.
    """

    name = "langsmith_metadata"

    def __init__(
        self,
        identity_provider: Callable[[], Mapping[str, Any] | None] | None = None,
    ) -> None:
        self._identity_provider = identity_provider or (lambda: None)

    def _stamp(self, ctx: CallContext) -> None:
        span = ctx.state.get("otel_span")
        if span is None:
            return
        metadata = self._identity_provider()
        if not metadata:
            return
        for key, value in metadata.items():
            if value is None:
                continue
            span.set_attribute(f"langsmith.metadata.{key}", str(value))

    def pre(self, ctx: CallContext) -> None:
        self._stamp(ctx)

    def post(self, ctx: CallContext, usage: UsageRecord) -> None:
        # No-op — metadata is set in pre so it lands before the span is
        # exported (in case an OTel batch flushes mid-call). Kept for
        # Hook Protocol compliance and to leave the door open for
        # response-derived metadata later.
        return
