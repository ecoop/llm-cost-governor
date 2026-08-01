# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""OTel processor + exporter wrapper for per-session telemetry control.

Two components, both consumed at OTel pipeline setup:

1. ``TelemetryModeStampingProcessor`` (a ``SpanProcessor``) reads the
   telemetry-mode baggage from each span's parent context at
   ``on_start`` and stamps the value as a span attribute. Baggage
   doesn't survive the span lifecycle on its own; copying it onto the
   span as an attribute carries the user's intent forward to the
   export stage where the actual content-scrubbing happens.

2. ``ContentScrubbingExporter`` wraps an underlying ``SpanExporter``.
   In ``export()`` it inspects each span's telemetry-mode attribute
   and, for ``"metadata"`` spans, returns a proxy with content
   attributes stripped before handing the sequence to the wrapped
   exporter. ``"full"`` spans pass through untouched; ``"off"`` should
   never reach here (the sampler drops them earlier), but is handled
   defensively.

Content attribute names follow the OTel GenAI semantic conventions
that ``opentelemetry-instrumentation-anthropic`` emits when
``TRACELOOP_TRACE_CONTENT=true``. The list below was verified by
reading the instrumentation's ``span_utils.py`` — if it adds new
content attributes upstream, this set should be extended.

Why a wrapping exporter rather than a custom processor:
    Processors get a writable span at ``on_start`` (already used by
    the stamping processor above) but a read-only span at ``on_end``.
    Mutating attributes at export time requires constructing a proxy
    around the ``ReadableSpan`` — same goal, different layer.
"""

from typing import Sequence

from opentelemetry import baggage as otel_baggage
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


# Attributes that carry prompt / response content. Stripped from spans
# whose telemetry_mode is "metadata". Sourced from
# opentelemetry-instrumentation-anthropic/span_utils.py — see this
# module's docstring for the audit trail.
CONTENT_ATTRIBUTES = frozenset({
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool_definitions",
    "gen_ai.request.structured_output.schema",
})

# The attribute name we stamp from baggage and read at export time.
# Kept as the historical Pitchcraft-flavored default so existing
# LangSmith queries and dashboards keep working; consumers building
# fresh integrations can pass their own value at instantiation.
MODE_ATTR = "pitchcraft.telemetry_mode"


class TelemetryModeStampingProcessor(SpanProcessor):
    """Stamps each span with the active telemetry-mode baggage value.

    Carries the user's mode choice from the request-time context onto
    the span itself, where it survives the lifecycle and reaches the
    export stage. Without this, the exporter would have no way to know
    a finalized span's telemetry_mode — baggage doesn't propagate
    automatically into ``ReadableSpan``.

    Args:
        mode_attr: The baggage key (and span-attribute name) to
            read + stamp. Defaults to the historical
            ``pitchcraft.telemetry_mode`` — override at instantiation
            for consumers who want a different namespace.
    """

    def __init__(self, mode_attr: str = MODE_ATTR) -> None:
        self._mode_attr = mode_attr

    def on_start(self, span, parent_context=None) -> None:
        mode = otel_baggage.get_baggage(self._mode_attr, parent_context)
        if mode:
            span.set_attribute(self._mode_attr, mode)

    def on_end(self, span: ReadableSpan) -> None:  # noqa: D401
        """No-op."""
        pass

    def shutdown(self) -> None:  # noqa: D401
        """No-op."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: D401
        """No-op; nothing buffered."""
        return True


class _ScrubbedSpan:
    """Proxy around a ``ReadableSpan`` that exposes a filtered attribute set.

    Delegates every attribute lookup to the wrapped span EXCEPT
    ``attributes``, which returns the scrubbed dict. The OTLP exporter
    reads ``span.attributes`` via ordinary attribute access, so a thin
    ``__getattr__`` proxy is sufficient — we don't need to deeply
    impersonate ``ReadableSpan``.
    """

    def __init__(self, wrapped: ReadableSpan, attributes: dict) -> None:
        self._wrapped = wrapped
        self._attributes = attributes

    @property
    def attributes(self) -> dict:
        return self._attributes

    def __getattr__(self, name: str):
        # Triggered only for attributes NOT on `self`, so delegating to
        # the wrapped span here doesn't shadow our `attributes` property
        # override above.
        return getattr(self._wrapped, name)


def scrub_span_content(
    span: ReadableSpan, *, mode_attr: str = MODE_ATTR
) -> ReadableSpan:
    """Return ``span`` (or a proxy) with content attributes removed if mode is metadata.

    Exposed for testing — the exporter calls this on each span before
    forwarding to its underlying exporter.

    Behavior by mode (from ``mode_attr`` on the span):

    - ``"metadata"`` → return proxy with ``CONTENT_ATTRIBUTES`` stripped.
    - ``"full"`` → return span unchanged.
    - missing / anything else → return span unchanged (defensive; the
      sampler should have dropped ``"off"`` spans earlier).
    """
    attrs = span.attributes or {}
    mode = attrs.get(mode_attr)
    if mode != "metadata":
        return span
    new_attrs = {k: v for k, v in attrs.items() if k not in CONTENT_ATTRIBUTES}
    return _ScrubbedSpan(span, new_attrs)


class ContentScrubbingExporter(SpanExporter):
    """SpanExporter wrapper that strips content from ``"metadata"`` spans.

    Composes with the real exporter (typically ``OTLPSpanExporter``) by
    intercepting ``export()``. Per-span scrubbing decisions happen in
    ``scrub_span_content``; this class is the SDK plumbing.

    Pass-through for shutdown and force_flush — they're cheap to
    forward and the underlying exporter owns the lifecycle.

    Args:
        wrapped: The concrete exporter to forward to (e.g. OTLP).
        mode_attr: The span-attribute name to read for the
            telemetry-mode decision. Must agree with the value used by
            the paired ``TelemetryModeStampingProcessor``.
    """

    def __init__(self, wrapped: SpanExporter, *, mode_attr: str = MODE_ATTR) -> None:
        self._wrapped = wrapped
        self._mode_attr = mode_attr

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        scrubbed = [scrub_span_content(s, mode_attr=self._mode_attr) for s in spans]
        return self._wrapped.export(scrubbed)

    def shutdown(self) -> None:
        self._wrapped.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._wrapped.force_flush(timeout_millis)
