# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Operator-alert seam — a swappable sink for cap-trip / write-failure notices.

The `llm_cost_governor` package needs to notify a human when things go
wrong (cap tripped, state write failed, unpriced model billed at $0),
but the *how* — Discord webhook, PagerDuty, Slack, email, stdout — is
an application concern. This module provides the seam:

    - `AlertSink` — the Protocol that any delivery implementation
      satisfies (a single `alert(level, title, body)` method).
    - `NoopAlertSink` — the default: silently drops every alert.
      Deliberate default so library code can call `alert(...)` without
      surprising a consumer who never registered a sink.
    - `set_alert_sink(sink)` — registers the process-wide sink (called
      once at app startup by the host app).
    - `alert(level, title, body)` — the module-level convenience the
      library uses. Routes to the current sink and *never raises* — a
      delivery failure is logged and swallowed so it can't break the
      request that triggered it.

Severity constants (INFO / WARNING / CRITICAL) match the shape the
Pitchcraft host app already uses, so consumers don't need to translate.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

_log = logging.getLogger(__name__)

# Severity levels. A closed set so a delivery layer can map each to a
# colour glyph / role mention / channel without parsing the title.
INFO = "info"
WARNING = "warning"
CRITICAL = "critical"


@runtime_checkable
class AlertSink(Protocol):
    """Delivers an operator alert somewhere.

    Implementations must not raise into the caller — the library's
    `alert()` wrapper also catches, but implementations should treat
    delivery as best-effort and never propagate failures.
    """

    def alert(self, level: str, title: str, body: str) -> None:
        """Deliver one alert; must never raise into the caller."""
        ...


class NoopAlertSink:
    """The default sink — silently drops every alert.

    Chosen so a consumer that never registers a real sink still sees a
    working library (silent alerts, but no crashes). Log-only variants
    (e.g. writing every alert to stdout) can be registered by the
    consumer if wanted.
    """

    def alert(self, level: str, title: str, body: str) -> None:
        return


_current: AlertSink = NoopAlertSink()


def set_alert_sink(sink: AlertSink) -> None:
    """Register the process-wide sink; called once at app startup.

    Overwrites any previous registration. Safe to call multiple times
    (last write wins), though production code should register once from
    a single startup path.
    """
    global _current
    _current = sink


def alert(level: str, title: str, body: str) -> None:
    """Route one alert to the current sink; never raises.

    The library-side entry point every llm-cost-governor module calls. Any
    exception from the sink is caught and logged; the caller never
    observes a delivery failure. This is intentional: an alert delivery
    problem must never break the response that triggered the alert.
    """
    try:
        _current.alert(level, title, body)
    except Exception:  # noqa: BLE001 — alerting must never break a response
        _log.exception("alert delivery failed [%s] %s", level, title)
