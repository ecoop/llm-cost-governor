# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Optional OpenTelemetry integration for llm_governor.

Importing anything from this subpackage transitively imports
``opentelemetry-*``; consumers that don't want OTel just don't import
from here and the core package stays OTel-free.

Modules:

    * ``scrubbing`` — ``TelemetryModeStampingProcessor`` +
      ``ContentScrubbingExporter``. Copies a per-request telemetry-mode
      baggage value onto each span at start, then strips prompt/response
      content attributes at export time for spans marked ``"metadata"``.

    * ``request_span`` — ``request_span(name, *, telemetry_mode,
      metadata, ...)`` context manager. Sets telemetry-mode baggage,
      opens a span, stamps caller-supplied metadata as
      ``langsmith.metadata.*`` attributes, becomes the parent of any
      descendant spans (including auto-instrumented provider spans).

    * ``hooks`` — ``OTelSpanHook`` + ``LangSmithMetadataHook``.
      ``Hook``-Protocol adapters that compose into a ``guarded_call``
      chain: the span hook opens a span in ``pre`` and closes it with
      token/cost attributes in ``post``; the metadata hook stamps a
      caller-supplied identity dict as ``langsmith.metadata.*``.
"""
