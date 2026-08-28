# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""OpenAI provider adapter — Chat Completions.

Implements the module-level adapter API `guarded_call` expects (see
``providers/__init__.py``): ``call``, ``extract_usage``, ``truncated``,
``default_estimate``.

**This adapter is a passthrough, not a translation layer.** ``call``
forwards the caller's kwargs to ``client.chat.completions.create``
unchanged, exactly as the Anthropic adapter forwards to
``client.messages.create``. Callers pass OpenAI-native kwargs
(``messages``, ``max_completion_tokens``, OpenAI-shaped ``tools``).

That is a deliberate scope line. Collapsing Anthropic-style ``system``
blocks, rewriting ``input_schema`` into ``function.parameters``, and
mapping ``tool_choice`` would make this library a cross-provider API
shim — a large, opinionated, and separately-versioned surface that has
nothing to do with metering spend. Applications that want one call site
across providers own that translation in their own dispatch seam, where
it can follow their prompt shapes rather than a lowest common
denominator this library would have to guess at.
"""

from __future__ import annotations

from typing import Any

from llm_cost_governor.wrapper import TokenEstimate

# Mirrors the Anthropic adapter's conservative pre-flight defaults: the real
# input size is unknowable before the call when caching is in play, and the
# output side is bounded by whatever ceiling the caller asked for.
_DEFAULT_INPUT_ESTIMATE = 2_000
_DEFAULT_MAX_TOKENS = 1_024


def call(client: Any, kwargs: dict) -> Any:
    """Invoke ``client.chat.completions.create`` with the caller's kwargs.

    Kept as a one-liner so tests can monkeypatch this module's ``call``
    attribute to intercept the actual SDK invocation.
    """
    return client.chat.completions.create(**kwargs)


def extract_usage(response: Any) -> dict:
    """Normalize an OpenAI ``response.usage`` into the wrapper's shape.

    OpenAI names the token fields differently from Anthropic
    (``prompt_tokens`` / ``completion_tokens``) and reports cache hits
    under ``prompt_tokens_details.cached_tokens``.

    Two mappings worth stating explicitly:

    * ``cache_creation_input_tokens`` is always ``None``. OpenAI's prompt
      caching is automatic and has no separate creation step to account
      for, so ``None`` ("provider doesn't expose this metric") is the
      honest value rather than ``0`` ("provider reported zero").
    * ``cached_tokens`` are reported by OpenAI *inside* ``prompt_tokens``,
      so they are subtracted out of ``input_tokens`` here. Leaving them in
      would bill the same tokens twice — once at the full input rate and
      again at the cached rate.
    """
    u = response.usage
    prompt = int(getattr(u, "prompt_tokens", 0) or 0)
    cached = _cached_tokens(u)
    return {
        # Uncached remainder — see the double-billing note above.
        "input_tokens": max(prompt - (cached or 0), 0),
        "output_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": None,
    }


def _cached_tokens(usage: Any) -> int | None:
    """Cache-hit prompt tokens, or None when the field isn't reported."""
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return None
    cached = (
        details.get("cached_tokens")
        if isinstance(details, dict)
        else getattr(details, "cached_tokens", None)
    )
    return None if cached is None else int(cached)


def truncated(response: Any) -> bool:
    """True when the response was cut off at the output-token ceiling."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    return getattr(choices[0], "finish_reason", None) == "length"


def default_estimate(kwargs: dict) -> TokenEstimate:
    """Conservative pre-flight token estimate from the caller's kwargs.

    Accepts either output-ceiling spelling: ``max_completion_tokens``
    (current) or ``max_tokens`` (legacy, still accepted by older models).
    """
    ceiling = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
    return TokenEstimate(
        input_tokens=_DEFAULT_INPUT_ESTIMATE,
        output_tokens=int(ceiling or _DEFAULT_MAX_TOKENS),
    )
