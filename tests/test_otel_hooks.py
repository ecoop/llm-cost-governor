# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Unit tests for llm_governor.otel.hooks + request_span.

Uses an in-memory OTel span exporter so tests observe the real span
attributes the hooks/request_span emit, not a mock.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_governor.otel.hooks import LangSmithMetadataHook, OTelSpanHook
from llm_governor.otel.request_span import request_span
from llm_governor.schemas import TokenEstimate, UsageRecord
from llm_governor.wrapper import CallContext

# ── Test infrastructure ──────────────────────────────────────────────────────


# OTel's global tracer_provider is set-once per process — trying to
# swap it per-test triggers a warning and recursion errors on teardown.
# Install one shared provider + in-memory exporter for the whole module;
# the per-test fixture just clears the exporter between tests.
_shared_exporter = InMemorySpanExporter()
_shared_provider = TracerProvider()
_shared_provider.add_span_processor(SimpleSpanProcessor(_shared_exporter))
try:
    trace.set_tracer_provider(_shared_provider)
except Exception:  # noqa: BLE001, S110 — another test module may have already set one
    pass


@pytest.fixture
def span_exporter():
    """Yield the module-shared in-memory exporter, cleared between tests."""
    _shared_exporter.clear()
    yield _shared_exporter


def _ctx(model: str = "claude-sonnet-4-6", tags: dict | None = None) -> CallContext:
    return CallContext(
        provider="anthropic",
        model=model,
        kwargs={"model": model},
        tags=tags or {},
        estimate=TokenEstimate(input_tokens=0, output_tokens=0),
    )


def _usage(model: str = "claude-sonnet-4-6", **overrides) -> UsageRecord:
    defaults = {
        "provider": "anthropic",
        "model": model,
        "input_tokens": 1_000,
        "output_tokens": 500,
        "cost_usd": 0.01,
    }
    defaults.update(overrides)
    return UsageRecord(**defaults)


# ── OTelSpanHook ─────────────────────────────────────────────────────────────


def test_otelspanhook_opens_span_in_pre_and_closes_in_post(span_exporter):
    hook = OTelSpanHook(tracer_name="test_hook")
    ctx = _ctx(model="claude-sonnet-4-6", tags={"stage": "drafter", "agent": "drafter_cv"})

    hook.pre(ctx)
    # Span object stashed for post handoff.
    assert "otel_span" in ctx.state
    # No spans exported yet — post hasn't run, span isn't finalized.
    assert span_exporter.get_finished_spans() == ()

    hook.post(ctx, _usage())

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "llm.anthropic.call"
    assert span.attributes["gen_ai.system"] == "anthropic"
    assert span.attributes["gen_ai.request.model"] == "claude-sonnet-4-6"
    # Tags passed through as llm_governor.tag.<key>.
    assert span.attributes["llm_governor.tag.stage"] == "drafter"
    assert span.attributes["llm_governor.tag.agent"] == "drafter_cv"
    # Usage stamped from the priced UsageRecord.
    assert span.attributes["gen_ai.usage.input_tokens"] == 1_000
    assert span.attributes["gen_ai.usage.output_tokens"] == 500
    assert span.attributes["llm_governor.cost_usd"] == pytest.approx(0.01)


def test_otelspanhook_stamps_cache_attrs_only_when_non_none(span_exporter):
    hook = OTelSpanHook(tracer_name="test_cache")
    ctx = _ctx()
    hook.pre(ctx)
    # `None` means "provider doesn't expose"; we should NOT emit a zero.
    hook.post(ctx, _usage(cache_read_input_tokens=None, cache_creation_input_tokens=None))

    (span,) = span_exporter.get_finished_spans()
    assert "gen_ai.usage.cache_read_input_tokens" not in span.attributes
    assert "gen_ai.usage.cache_creation_input_tokens" not in span.attributes


def test_otelspanhook_stamps_cache_attrs_when_zero(span_exporter):
    hook = OTelSpanHook(tracer_name="test_cache_zero")
    ctx = _ctx()
    hook.pre(ctx)
    # `0` means "provider reported zero" — distinct from missing.
    hook.post(ctx, _usage(cache_read_input_tokens=0, cache_creation_input_tokens=0))

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes["gen_ai.usage.cache_read_input_tokens"] == 0
    assert span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 0


def test_otelspanhook_post_is_noop_when_span_missing():
    """If ctx.state has no 'otel_span' key, post must not raise."""
    hook = OTelSpanHook(tracer_name="test_missing")
    ctx = _ctx()
    # Don't call pre — ctx.state stays empty.
    hook.post(ctx, _usage())  # must not raise


# ── LangSmithMetadataHook ────────────────────────────────────────────────────


def test_langsmithmetadatahook_stamps_provider_dict_on_span(span_exporter):
    span_hook = OTelSpanHook(tracer_name="test_meta")
    meta_hook = LangSmithMetadataHook(
        identity_provider=lambda: {
            "session_id": "sess-abc",
            "user_id": "01KABC",
            "persona": "eric",
            "invite_token": None,  # skipped
        }
    )
    ctx = _ctx()

    span_hook.pre(ctx)
    meta_hook.pre(ctx)
    span_hook.post(ctx, _usage())

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes["langsmith.metadata.session_id"] == "sess-abc"
    assert span.attributes["langsmith.metadata.user_id"] == "01KABC"
    assert span.attributes["langsmith.metadata.persona"] == "eric"
    assert "langsmith.metadata.invite_token" not in span.attributes


def test_langsmithmetadatahook_no_provider_is_noop(span_exporter):
    """With no identity_provider (default), the hook adds nothing."""
    span_hook = OTelSpanHook(tracer_name="test_meta_none")
    meta_hook = LangSmithMetadataHook()  # default provider returns None
    ctx = _ctx()

    span_hook.pre(ctx)
    meta_hook.pre(ctx)
    span_hook.post(ctx, _usage())

    (span,) = span_exporter.get_finished_spans()
    assert not any(k.startswith("langsmith.metadata.") for k in span.attributes)


def test_langsmithmetadatahook_pre_without_span_is_noop():
    """Metadata hook must not raise when no upstream span-hook ran."""
    meta_hook = LangSmithMetadataHook(identity_provider=lambda: {"user_id": "01K"})
    ctx = _ctx()
    meta_hook.pre(ctx)  # ctx.state has no 'otel_span'; must not raise


# ── request_span ─────────────────────────────────────────────────────────────


def test_request_span_stamps_metadata_and_returns_active_span(span_exporter):
    with request_span(
        "workflow_test",
        telemetry_mode="metadata",
        metadata={"session_id": "s1", "persona": "eric", "extra": None},
        tracer_name="test_reqspan",
    ) as span:
        # Yielded span is the currently-active span.
        assert trace.get_current_span() is span

    (finished,) = span_exporter.get_finished_spans()
    assert finished.name == "workflow_test"
    assert finished.attributes["langsmith.metadata.session_id"] == "s1"
    assert finished.attributes["langsmith.metadata.persona"] == "eric"
    # None values skipped.
    assert "langsmith.metadata.extra" not in finished.attributes


def test_request_span_children_inherit_baggage_via_active_context(span_exporter):
    """Descendant spans opened inside request_span become its children."""
    tracer = trace.get_tracer("test_child")
    with (
        request_span("outer", telemetry_mode="full", tracer_name="test_reqspan_child") as outer,
        tracer.start_as_current_span("inner"),
    ):
        pass  # inner closes here

    spans_by_name = {s.name: s for s in span_exporter.get_finished_spans()}
    assert set(spans_by_name) == {"outer", "inner"}
    # inner's parent is outer.
    assert spans_by_name["inner"].parent.span_id == outer.get_span_context().span_id


def test_request_span_default_telemetry_mode_is_off():
    """Fail-safe default: caller who forgets `telemetry_mode` gets 'off'."""
    from opentelemetry import baggage
    # Enter the CM and check what baggage is active mid-span.
    with request_span("check_default", tracer_name="test_default"):
        # Default baggage key = "pitchcraft.telemetry_mode" (historical).
        assert baggage.get_baggage("pitchcraft.telemetry_mode") == "off"
