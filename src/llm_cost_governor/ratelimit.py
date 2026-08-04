# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Per-client request rate limiter — framework-neutral core.

The request-rate complement to the LLM spend counter (which caps spend,
not request rate). Where the cost counter needs the LLM ``usage`` object,
this limiter needs the client identity — typically a client IP.

A single module-level ``ip_rate_limiter`` singleton tracks request counts
per client in a fixed 60-second window and raises ``RateLimitExceeded``
when the caller exceeds the given per-window limit.

State model:
    In-memory only — no persistence. A restart resets every window,
    which is acceptable: the limiter exists to blunt bursts within a
    minute, and the durable spend caps in ``counters`` are the real
    backstop for cumulative abuse. Fixed-window (not sliding) for
    simplicity: each caller's count resets when its 60-second window
    rolls over, so a determined client can send up to 2×limit across a
    window boundary.

Single-instance assumption:
    Correct only at max-instances=1 (no cross-instance shared counter).
    Multi-instance correctness is out of scope for this base.

Framework-neutral by design:
    This module owns the counter and the exception; the FastAPI
    dependency + client-IP extractor live in ``llm_cost_governor.fastapi_ext``
    so a non-FastAPI caller can use the limiter directly.
"""

from __future__ import annotations

import threading
import time

_WINDOW_SECONDS = 60.0


# ── Exception ─────────────────────────────────────────────────────────────────


class RateLimitExceeded(Exception):
    """Raised when a client exceeds its per-window request budget.

    Attributes:
        retry_after: Whole seconds the client should wait before retrying
            — the time left in the current window. Surface as the HTTP
            ``Retry-After`` header at the API layer.
        scope: Machine-readable limiter id — ``"ip"`` by default. Lets
            the API layer / alerting branch on which limiter tripped
            without parsing the message, mirroring ``CostCapExceeded.cap``.

    The message is user-facing; render as an HTTP 429 body.
    """

    def __init__(self, retry_after: int, message: str, scope: str = "ip") -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.scope = scope


# ── Limiter ───────────────────────────────────────────────────────────────────


class IPRateLimiter:
    """In-memory fixed-window request counter keyed by client identifier.

    Thread-safe: a single lock guards the per-caller window map, since
    ``check`` is called from the threadpool that runs sync endpoints
    under concurrent requests.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # ip → (window_start_monotonic, count_in_window)
        self._windows: dict[str, tuple[float, int]] = {}

    def check(self, ip: str, limit: int) -> None:
        """Count one request for ``ip``; raise if it exceeds ``limit``.

        Fixed 60-second window per caller: the first request opens a
        window; subsequent requests increment the count until the window
        rolls over (60s after it opened), at which point the count
        resets.

        Args:
            ip: Client identifier (e.g. an already-extracted IP).
            limit: Max requests allowed within the window.

        Raises:
            RateLimitExceeded: When this request would push the count
                over ``limit``. ``retry_after`` is the seconds left in
                the current window.
        """
        now = time.monotonic()
        with self._lock:
            window_start, count = self._windows.get(ip, (now, 0))
            elapsed = now - window_start
            if elapsed >= _WINDOW_SECONDS:
                # Window rolled over — start a fresh one for this request.
                window_start, count = now, 0
            count += 1
            if count > limit:
                retry_after = max(1, int(_WINDOW_SECONDS - elapsed) + 1)
                # Don't record the rejected request; leave the existing
                # window intact so the count reflects only served requests.
                raise RateLimitExceeded(
                    retry_after,
                    "Too many requests. The demo limits how fast calls can be "
                    f"made (up to {limit} per minute). Please wait about "
                    f"{retry_after}s and try again.",
                )
            self._windows[ip] = (window_start, count)

    def reset(self) -> None:
        """Clear all windows. Test helper — not used in production."""
        with self._lock:
            self._windows.clear()


# Module-level singleton — imported by the LLM POST endpoint dependencies.
ip_rate_limiter = IPRateLimiter()
