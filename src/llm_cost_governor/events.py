# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Per-LLM-call event log — one structured JSON line per real LLM call.

A companion to the rolling-window cost counter (``llm_cost_governor.counters``).
The counter is sized for one job — fast cap enforcement on the hot path
— and deliberately throws away per-call detail (per-model spend,
prompt-cache hit rate, per-stage attribution, per-run reconstruction).
This module captures that rich detail without bloating the counter:
**one structured JSON line per real LLM call**, emitted to stdout.

Sink — stdout → Cloud Logging:
    GCS objects have no atomic append (concurrent writers race, and
    the debounced-rewrite trick doesn't fit an append-only log),
    whereas on Cloud Run anything written to stdout is captured by
    Cloud Logging automatically — queryable with ``jq``-like filters,
    no extra bucket, IAM, or write path. A single-line JSON object is
    ingested as a structured ``jsonPayload``.

Gating:
    Every emit takes an ``enabled`` argument. Off → cheap no-op. The
    caller is responsible for deciding when to enable (typically wired
    from an app-level flag at startup).

Privacy:
    The raw invite token (or any credential) is never logged
    directly. Callers pass an ``identity`` dict with a ``"token"`` key;
    this module logs ``sha256(token)`` — correlatable across calls and
    joinable to an identity map that keys on the same hash — plus any
    other identity fields (``"recipient"``, etc.) verbatim.

Robustness:
    Fire-and-forget: a logging failure must never break the response
    the emit is attached to, so everything is wrapped and swallowed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from llm_cost_governor.schemas import UsageRecord
from llm_cost_governor.wrapper import CallContext

_log = logging.getLogger(__name__)


def _token_hash(token: str) -> str:
    """sha256 hex of a credential string."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def emit_llm_call(
    *,
    enabled: bool = False,
    stage: str = "",
    agent: str | None = None,
    usage: Mapping[str, Any],
    usd: float,
    identity: Mapping[str, Any] | None = None,
    session_id: str | None = None,
) -> None:
    """Emit one structured event line for a completed LLM call.

    A no-op unless ``enabled`` is True. Never raises — any failure is
    logged and swallowed so the caller's response is unaffected.

    Args:
        enabled: Master gate. When False, this function returns
            immediately. Caller decides the policy.
        stage: Human-readable calling-stage name (e.g. ``"drafter"``).
        agent: The agent slug (e.g. ``"critic_cv"``), or None.
        usage: An Anthropic-style ``usage`` dict — ``model`` /
            ``input_tokens`` / ``output_tokens`` /
            ``cache_read_input_tokens`` / ``cache_creation_input_tokens``.
        usd: The USD cost of this call.
        identity: Optional caller identity dict. When present, its
            ``"token"`` key (if any) is hashed with sha256 and logged
            as ``token_hash`` (the raw token never appears in the
            payload); its ``"recipient"`` key (if any) is logged
            verbatim as ``recipient``.
        session_id: Optional session/request id, logged as ``session``.
    """
    if not enabled:
        return

    try:
        token = identity.get("token") if identity else None
        recipient = identity.get("recipient") if identity else None
        payload = {
            "event": "llm_call",
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session": session_id,
            "token_hash": _token_hash(token) if token else None,
            "recipient": recipient,
            "stage": stage,
            "agent": agent,
            "model": usage.get("model", ""),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation": usage.get("cache_creation_input_tokens", 0),
            "cache_read": usage.get("cache_read_input_tokens", 0),
            "usd": round(usd, 6),
        }
        # One compact JSON object per line → Cloud Logging structured payload.
        print(json.dumps(payload, separators=(",", ":")), file=sys.stdout, flush=True)
    except Exception:  # noqa: BLE001 — event logging must never break a response
        _log.exception("event_log: failed to emit LLM-call event.")


# ── Hook adapter for guarded_call ──────────────────────────────────────────────


class EventLogHook:
    """`Hook` adapter that emits one event-log line per completed call.

    `post` pulls ``stage`` / ``agent`` from ``ctx.tags`` and calls the
    caller-supplied identity + session-id providers to gather the
    rest of the metadata, then delegates to :func:`emit_llm_call`.

    Both providers default to a no-op — the hook composes cleanly for
    anonymous apps (no auth, no session tracking) as well as for
    authenticated apps that thread identity through.

    `pre` is a no-op — event logging happens after the call.
    """

    name = "event_log"

    def __init__(
        self,
        *,
        enabled: bool = False,
        identity_provider: Callable[[], Mapping[str, Any] | None] | None = None,
        session_id_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self._enabled = enabled
        self._identity_provider = identity_provider or (lambda: None)
        self._session_id_provider = session_id_provider or (lambda: None)

    def pre(self, ctx: CallContext) -> None:
        return

    def post(self, ctx: CallContext, usage: UsageRecord) -> None:
        emit_llm_call(
            enabled=self._enabled,
            stage=ctx.tags.get("stage", ""),
            agent=ctx.tags.get("agent"),
            usage={
                "model": usage.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            },
            usd=usage.cost_usd,
            identity=self._identity_provider(),
            session_id=self._session_id_provider(),
        )
