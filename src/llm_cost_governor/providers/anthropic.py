# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Anthropic provider adapter.

Wraps ``client.messages.create`` and normalizes the response.usage
shape into the four-field dict the wrapper + pricing table consume.
The 8k default input estimate for pre-flight budget checks matches the
pre-extraction ``safe_create`` heuristic.
"""

from __future__ import annotations

from typing import Any

from llm_cost_governor.schemas import TokenEstimate

# Conservative pre-flight input estimate used when the caller doesn't
# supply their own — matches the pre-extraction `_INPUT_ESTIMATE` in
# `agents/_anthropic_helpers.py`. Errs on the side of caution for
# cache-heavy prompts.
_DEFAULT_INPUT_ESTIMATE = 8_000

# Anthropic's `max_tokens` param has no soft default — the SDK requires
# a value. This fallback exists only for the estimate path when a
# caller somehow omitted it.
_DEFAULT_MAX_TOKENS = 4096


def call(client: Any, kwargs: dict) -> Any:
    """Invoke ``client.messages.create`` with the caller's kwargs.

    Kept as a one-liner so tests can monkeypatch this module's ``call``
    attribute to intercept the actual SDK invocation.
    """
    return client.messages.create(**kwargs)


def extract_usage(response: Any) -> dict:
    """Normalize the Anthropic response.usage into the wrapper's shape.

    Returns a dict with the four token fields — cache values are
    ``None`` when the response doesn't include them (older API versions
    or non-cached calls), so downstream ``UsageRecord`` construction
    distinguishes "provider didn't expose" from "provider reported 0".
    """
    u = response.usage
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
        "server_tool_use": _server_tool_use(u),
    }


def _server_tool_use(usage: Any) -> dict[str, int] | None:
    """Normalize ``usage.server_tool_use`` to a plain ``{key: count}`` dict.

    Anthropic bills some server-side tool use on top of tokens (web search
    at $10/1,000 searches) and reports it here, alongside the token counts.
    Returned as a plain dict rather than the SDK object so nothing
    downstream depends on the SDK's shape.

    Returns None when the call made no server-tool use, matching the
    cache fields' "provider didn't report this" convention.
    """
    stu = getattr(usage, "server_tool_use", None)
    if stu is None:
        return None
    if not isinstance(stu, dict):
        stu = getattr(stu, "__dict__", None) or {
            k: getattr(stu, k) for k in dir(stu)
            if k.endswith("_requests") and not k.startswith("_")
        }
    counts = {k: int(v) for k, v in stu.items() if isinstance(v, int | float)}
    return counts or None


def truncated(response: Any) -> bool:
    """True when the Anthropic response was cut off at ``max_tokens``."""
    return getattr(response, "stop_reason", None) == "max_tokens"


def default_estimate(kwargs: dict) -> TokenEstimate:
    """Conservative pre-flight token estimate from the caller's kwargs.

    Uses a fixed input estimate (unreliable to compute from the prompt
    when prompt caching is in play) plus the caller's requested
    ``max_tokens`` for the output side.
    """
    return TokenEstimate(
        input_tokens=_DEFAULT_INPUT_ESTIMATE,
        output_tokens=int(kwargs.get("max_tokens") or _DEFAULT_MAX_TOKENS),
    )
