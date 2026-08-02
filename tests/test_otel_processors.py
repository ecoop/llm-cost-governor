# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Unit tests for the OTel processor + exporter wrapper (#197).

Covers the content-scrubbing logic in isolation — no real OTel SDK
spans or exporters required, just dict-attributes spans built with
``SimpleNamespace`` (close enough to a ReadableSpan for the export
codepath we care about).
"""

from collections.abc import Sequence
from types import SimpleNamespace

from llm_guardrails.otel.scrubbing import (
    CONTENT_ATTRIBUTES,
    MODE_ATTR,
    ContentScrubbingExporter,
    _ScrubbedSpan,
    scrub_span_content,
)


def _fake_span(**attrs) -> SimpleNamespace:
    """Build a thing that quacks like a ReadableSpan for our exporter."""
    return SimpleNamespace(
        name="test_span",
        attributes=attrs,
        # other fields the OTLP exporter would read; minimal here since
        # our wrapper doesn't inspect them, only the underlying exporter
        # does (mocked below).
    )


# ── scrub_span_content ───────────────────────────────────────────────────────


def test_scrub_returns_span_unchanged_when_mode_is_full():
    span = _fake_span(
        **{
            MODE_ATTR: "full",
            "gen_ai.input.messages": "secret prompt",
            "gen_ai.usage.input_tokens": 100,
        }
    )
    result = scrub_span_content(span)
    assert result is span  # identity, not a copy
    assert result.attributes["gen_ai.input.messages"] == "secret prompt"


def test_scrub_strips_content_when_mode_is_metadata():
    span = _fake_span(
        **{
            MODE_ATTR: "metadata",
            "gen_ai.input.messages": "secret prompt",
            "gen_ai.output.messages": "secret response",
            "gen_ai.system_instructions": "secret system",
            "gen_ai.tool_definitions": "secret tools",
            "gen_ai.request.structured_output.schema": "secret schema",
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.request.model": "claude-sonnet-4-6",
        }
    )
    result = scrub_span_content(span)
    # Content attrs gone
    for attr in (
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool_definitions",
        "gen_ai.request.structured_output.schema",
    ):
        assert attr not in result.attributes
    # Metadata attrs preserved
    assert result.attributes["gen_ai.usage.input_tokens"] == 100
    assert result.attributes["gen_ai.request.model"] == "claude-sonnet-4-6"
    assert result.attributes[MODE_ATTR] == "metadata"


def test_scrub_passes_through_when_mode_attr_missing():
    """Defensive: spans without our mode attr are unchanged (shouldn't happen
    in practice — workflow_span always sets it — but if it does we don't
    silently drop content)."""
    span = _fake_span(**{"gen_ai.input.messages": "preserved"})
    result = scrub_span_content(span)
    assert result is span
    assert result.attributes["gen_ai.input.messages"] == "preserved"


def test_scrub_passes_through_when_mode_is_off_or_unknown():
    """Sampler should drop "off" before export, but defensive: passes through."""
    for mode in ("off", "unknown_value", "", None):
        span = _fake_span(
            **{MODE_ATTR: mode, "gen_ai.input.messages": "preserved"}
        )
        result = scrub_span_content(span)
        # Either same object (no scrub) or proxy with content intact
        assert "gen_ai.input.messages" in result.attributes


def test_scrub_handles_span_with_no_attributes():
    span = SimpleNamespace(name="empty", attributes=None)
    # Should not raise
    result = scrub_span_content(span)
    assert result is span


# ── _ScrubbedSpan proxy ──────────────────────────────────────────────────────


def test_scrubbed_span_delegates_non_attribute_lookups():
    original = SimpleNamespace(
        name="original",
        kind="INTERNAL",
        attributes={"a": 1, "b": 2},
    )
    proxy = _ScrubbedSpan(original, {"a": 1})
    # attributes is overridden
    assert proxy.attributes == {"a": 1}
    # Other lookups delegate to the wrapped span
    assert proxy.name == "original"
    assert proxy.kind == "INTERNAL"


# ── ContentScrubbingExporter ─────────────────────────────────────────────────


class _RecordingExporter:
    """Mock SpanExporter that records what it was asked to export."""

    def __init__(self) -> None:
        self.exported: list = []
        self.shutdown_called = False
        self.flush_called = False

    def export(self, spans: Sequence) -> object:
        self.exported.append(list(spans))
        return "SUCCESS"

    def shutdown(self) -> None:
        self.shutdown_called = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self.flush_called = True
        return True


def test_exporter_strips_metadata_spans_before_forwarding():
    inner = _RecordingExporter()
    exporter = ContentScrubbingExporter(inner)
    spans = [
        _fake_span(
            **{MODE_ATTR: "metadata", "gen_ai.input.messages": "leaked?"}
        ),
        _fake_span(**{MODE_ATTR: "full", "gen_ai.input.messages": "kept"}),
    ]
    exporter.export(spans)
    forwarded = inner.exported[0]
    assert len(forwarded) == 2
    # Metadata span: content stripped
    assert "gen_ai.input.messages" not in forwarded[0].attributes
    # Full span: content preserved
    assert forwarded[1].attributes["gen_ai.input.messages"] == "kept"


def test_exporter_shutdown_forwards():
    inner = _RecordingExporter()
    ContentScrubbingExporter(inner).shutdown()
    assert inner.shutdown_called


def test_exporter_force_flush_forwards():
    inner = _RecordingExporter()
    assert ContentScrubbingExporter(inner).force_flush() is True
    assert inner.flush_called


# ── CONTENT_ATTRIBUTES audit ─────────────────────────────────────────────────


def test_content_attributes_covers_expected_gen_ai_keys():
    """Spot-check that the attribute list matches the documented set.

    If opentelemetry-instrumentation-anthropic adds a new content
    attribute upstream, this set should be extended in
    api/otel_processors.py and this test updated alongside.
    """
    expected = {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool_definitions",
        "gen_ai.request.structured_output.schema",
    }
    assert CONTENT_ATTRIBUTES == expected
