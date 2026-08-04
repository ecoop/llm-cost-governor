# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Generic per-request OTel span with consent baggage + metadata.

A context manager for wrapping a request-scoped unit of work in one
top-level span that:

  * Sets a telemetry-mode baggage value on the active context so that
    a ``ConsentSampler`` (up-pipeline) can drop the whole span subtree
    when the caller has opted out, and a ``ContentScrubbingExporter``
    (down-pipeline) can strip content attrs for ``"metadata"`` mode.
  * Opens a named span; caller-supplied ``metadata`` becomes
    ``langsmith.metadata.<key>`` attributes on it.
  * Becomes the parent of every span created inside the ``with``
    block — including auto-instrumented provider spans — restoring
    nested traces in LangSmith / any OTel-aware backend.

App-neutral: no dependency on Pitchcraft's ``Session`` / identity /
invite-token modules. Callers gather the metadata dict themselves and
pass it in. See ``api.observability.workflow_span`` for the
Pitchcraft-specific adapter.

Why a ``context=`` kwarg on the span (not manual attach/detach at the
call site): ``start_as_current_span(name, context=ctx)`` would only
propagate baggage to the sampling decision for the workflow span
itself — children would see no baggage in their ``parent_context``
and any consent-aware sampler would drop them. Instead we
``attach()`` the baggage-bearing context so descendant spans inherit
it via the active context; the ``finally`` block detaches.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import baggage as otel_baggage
from opentelemetry import context as otel_context
from opentelemetry import trace

# Default baggage key. Matches the historical Pitchcraft value so
# existing sampler/exporter configurations continue to work when a
# consumer supplies no override. Fresh integrations can pass their own.
DEFAULT_BAGGAGE_KEY = "pitchcraft.telemetry_mode"

# Default tracer name; only affects OTel introspection ("which library
# emitted this span"), not span content.
DEFAULT_TRACER_NAME = "llm_cost_governor"


@contextmanager
def request_span(
    name: str,
    *,
    telemetry_mode: str = "off",
    metadata: Mapping[str, Any] | None = None,
    baggage_key: str = DEFAULT_BAGGAGE_KEY,
    tracer_name: str = DEFAULT_TRACER_NAME,
) -> Iterator[trace.Span]:
    """Open a request-scoped span carrying telemetry-mode baggage + metadata.

    Args:
        name: Span name, e.g. ``"workflow_draft"``, ``"workflow_ask"``.
        telemetry_mode: The consent value to publish on baggage —
            typically ``"off"``, ``"metadata"``, or ``"full"``. A
            ``ConsentSampler`` reads this to decide whether to record
            the subtree; a ``ContentScrubbingExporter`` reads it to
            decide whether to strip content attrs at export time.
            Defaults to ``"off"`` so a caller who forgets the arg
            fails safe (opted out).
        metadata: Flat ``str → value`` mapping stamped on the span as
            ``langsmith.metadata.<key>`` attributes. ``None`` values
            are skipped so callers can pass optional fields without
            filtering first. Non-string values are ``str()``-coerced
            to match LangSmith's attribute-serialization expectations.
        baggage_key: The baggage key used to publish
            ``telemetry_mode`` on the active context. Must agree with
            whatever the sampler / exporter read (see the ``scrubbing``
            module's ``MODE_ATTR``).
        tracer_name: Name for ``trace.get_tracer(...)``. Consumers
            usually want their own application namespace here.

    Yields:
        The active ``trace.Span``. Callers usually don't need to
        interact with it directly — descendant spans become its
        children automatically via the active context.
    """
    ctx = otel_baggage.set_baggage(baggage_key, telemetry_mode)
    token = otel_context.attach(ctx)
    try:
        tracer = trace.get_tracer(tracer_name)
        with tracer.start_as_current_span(name) as span:
            if metadata:
                for key, value in metadata.items():
                    if value is None:
                        continue
                    span.set_attribute(f"langsmith.metadata.{key}", str(value))
            yield span
    finally:
        otel_context.detach(token)
