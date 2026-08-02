# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""FastAPI adapters for llm_guardrails — currently the IP-rate-limit dependency.

Optional integration layer: kept in a separate module so the core
guardrails package doesn't require FastAPI as a hard dependency.
Import from here only when wiring an actual FastAPI app.

Zero coupling to any host application: every value the adapters need
(rate limiter, cap value, enabled gate) is a factory arg. The caller
constructs the dependency once at startup and hands the result to
FastAPI's ``Depends()``.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Request

from llm_guardrails.ratelimit import IPRateLimiter

# ── Client-IP extraction ───────────────────────────────────────────────────────


def client_ip(request: Request) -> str:
    """Best-effort real client IP, honoring the front proxy's XFF header.

    Cloud Run sits behind an HTTPS-terminating proxy, so the socket
    peer is the proxy, not the visitor. ``X-Forwarded-For`` is a
    comma-list of ``client, proxy1, proxy2, ...``; the left-most
    entry is the original client. Falls back to the direct socket
    peer when no XFF is present (local dev, direct connections).

    Note:
        XFF is client-supplied and therefore spoofable in general.
        Behind a managed proxy the header is set by trusted
        infrastructure, so the left-most entry is good enough. Not a
        security boundary.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client is not None:
        return request.client.host
    return "unknown"


# ── FastAPI dependency factory ─────────────────────────────────────────────────


def make_enforce_ip_rate_limit(
    rate_limiter: IPRateLimiter,
    *,
    cap_rpm: int,
    enabled: bool = False,
) -> Callable[[Request], None]:
    """Build a FastAPI dependency that enforces a per-IP request rate limit.

    Call this once at app startup with the wired-up ``rate_limiter``
    instance + config values, then attach the returned callable to
    endpoints with ``dependencies=[Depends(dependency)]``.

    Args:
        rate_limiter: The ``IPRateLimiter`` instance to enforce
            against (typically one shared singleton per app).
        cap_rpm: Max requests per minute per client IP.
        enabled: When False, the returned dependency is a no-op. Wire
            this from your app's demo/production gate.

    Returns:
        A FastAPI-compatible dependency: a ``(request) -> None``
        callable that raises ``RateLimitExceeded`` (→ HTTP 429) when
        the caller's IP has crossed the cap.
    """

    def _enforce(request: Request) -> None:
        if not enabled:
            return
        rate_limiter.check(client_ip(request), cap_rpm)

    return _enforce
