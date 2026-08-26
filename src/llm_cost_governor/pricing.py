# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Per-model pricing table and cost math.

The model registry (`MODEL_PRICING`) is the single source of truth for
every model's display label, `capability` (`chat` / `embedding` /
`reranker`), and four USD-per-1M-token rates (`input`, `output`,
`cache_write`, `cache_read`). `RATES` is the cost-math view, projecting
exactly `RATE_KEYS` so descriptive fields can never leak into it, and
derived here so it can never drift from the registry.

`catalog()` returns the registry as typed `ModelRecord`s, optionally
filtered by capability — so callers can ask for "the chat models"
without re-deriving that from provider-name prefixes.

`_cost` and `usd_for_usage` price completed calls. Unpriced models are
tolerated (they cost $0 rather than raising) and pinged once per process
via `_warn_unpriced`.

Rates as of 2026-08-01:
    Claude:  https://www.anthropic.com/pricing
    Voyage:  https://docs.voyageai.com/docs/pricing
All figures in USD per 1 million tokens.

Voyage embedding and rerank models have no output / cache tokens —
their rows set ``output``, ``cache_write``, and ``cache_read`` to
0.0 so the existing ``_cost`` arithmetic works unchanged (input
tokens × input rate + zeros for the other terms).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

_log = logging.getLogger(__name__)

# ── Model registry (single source of truth) ────────────────────────────────────
# Keys are BARE model aliases (`claude-haiku-4-5`), never dated snapshots
# (`claude-haiku-4-5-20251001`). Bare aliases are how callers are meant to
# name current models — they float to the latest snapshot — so keying on
# them is what makes `_cost` resolve for ordinary application code. A dated
# key here silently bills that model at $0 (see #5). Enforced by
# `test_model_pricing_keys_are_bare_aliases`.
#
# Each row: `label` (UI display) + four USD-per-1M-token rates:
#   input        — uncached input tokens
#   output       — output tokens
#   cache_write  — cost to create a prompt-cache entry (1.25× input rate)
#   cache_read   — cost to read a cached entry         (0.10× input rate)

MODEL_PRICING: dict[str, dict] = {
    # ── Claude 5 family ──
    # Fable 5: Anthropic's most capable widely released model. Priced above
    # Opus tier for the most demanding long-horizon agentic and reasoning
    # workloads.
    "claude-fable-5": {
        "label": "Fable 5",
        "capability": "chat",
        "input":      10.00,
        "output":     50.00,
        "cache_write": 12.50,
        "cache_read":   1.00,
    },
    # Opus 5: current-generation Opus, same $5/$25 sticker as the 4.x
    # Opus line.
    "claude-opus-5": {
        "label": "Opus 5",
        "capability": "chat",
        "input":       5.00,
        "output":     25.00,
        "cache_write": 6.25,
        "cache_read":  0.50,
    },
    # Sonnet 5: current-generation Sonnet. Standard rate $3/$15; Anthropic
    # is running an introductory $2/$10 through 2026-08-31. We price at the
    # standard sticker — the intro discount is a billing-time credit and
    # doesn't need to be reflected in pre-flight budget math.
    "claude-sonnet-5": {
        "label": "Sonnet 5",
        "capability": "chat",
        "input":       3.00,
        "output":     15.00,
        "cache_write": 3.75,
        "cache_read":  0.30,
    },
    # ── Claude 4.x family (still active) ──
    "claude-sonnet-4-6": {
        "label": "Sonnet 4.6",
        "capability": "chat",
        "input":       3.00,
        "output":     15.00,
        "cache_write": 3.75,
        "cache_read":  0.30,
    },
    # Opus 4.6, 4.7, and 4.8 share the same per-token rate ($5/$25). Opus
    # 4.7+ use a new tokenizer that produces up to 35% more tokens from the
    # same text, so effective cost per request is higher — especially for
    # structured/code content.
    "claude-opus-4-6": {
        "label": "Opus 4.6",
        "capability": "chat",
        "input":       5.00,
        "output":     25.00,
        "cache_write": 6.25,
        "cache_read":  0.50,
    },
    "claude-opus-4-7": {
        "label": "Opus 4.7",
        "capability": "chat",
        "input":       5.00,
        "output":     25.00,
        "cache_write": 6.25,
        "cache_read":  0.50,
    },
    "claude-opus-4-8": {
        "label": "Opus 4.8",
        "capability": "chat",
        "input":       5.00,
        "output":     25.00,
        "cache_write": 6.25,
        "cache_read":  0.50,
    },
    # Haiku 4.5: cheapest current-generation Claude. Keyed bare, like every
    # other row — callers are meant to name the floating alias, not a dated
    # snapshot (see test_model_pricing_keys_are_bare_aliases).
    "claude-haiku-4-5": {
        "label": "Haiku 4.5",
        "capability": "chat",
        "input":       1.00,
        "output":      5.00,
        "cache_write": 1.25,
        "cache_read":  0.10,
    },

    # ── Voyage embeddings (current generation) ──
    # Embeddings have no output/cache dimensions — those fields are 0
    # so the same `_cost` arithmetic works for both providers. All rows
    # come with a 200M-token free tier unless noted; free tokens are
    # billing-side and not modeled here.
    "voyage-4": {
        "label": "Voyage 4",
        "capability": "embedding",
        "input":       0.06,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "voyage-4-large": {
        "label": "Voyage 4 Large",
        "capability": "embedding",
        "input":       0.12,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "voyage-4-lite": {
        "label": "Voyage 4 Lite",
        "capability": "embedding",
        "input":       0.02,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "voyage-context-4": {
        "label": "Voyage Context 4",
        "capability": "embedding",
        "input":       0.18,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "voyage-code-3": {
        "label": "Voyage Code 3",
        "capability": "embedding",
        "input":       0.18,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    # voyage-finance-2, voyage-law-2, voyage-code-2 share the same
    # $0.12/1M rate but a smaller 50M free tier.
    "voyage-finance-2": {
        "label": "Voyage Finance 2",
        "capability": "embedding",
        "input":       0.12,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "voyage-law-2": {
        "label": "Voyage Law 2",
        "capability": "embedding",
        "input":       0.12,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "voyage-code-2": {
        "label": "Voyage Code 2",
        "capability": "embedding",
        "input":       0.12,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },

    # ── Voyage rerank ──
    # Batch API applies a 33% discount, billed via the batch endpoint;
    # not modeled here since it's a request-shape distinction rather
    # than a model-id one.
    "rerank-2.5": {
        "label": "Voyage Rerank 2.5",
        "capability": "reranker",
        "input":       0.05,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "rerank-2.5-lite": {
        "label": "Voyage Rerank 2.5 Lite",
        "capability": "reranker",
        "input":       0.02,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "rerank-2": {
        "label": "Voyage Rerank 2",
        "capability": "reranker",
        "input":       0.05,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "rerank-2-lite": {
        "label": "Voyage Rerank 2 Lite",
        "capability": "reranker",
        "input":       0.02,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
}

# The four USD-per-1M-token rate keys. `RATES` projects exactly these — an
# allowlist, not "everything except `label`", so descriptive fields added to
# the registry (`capability`, and whatever comes next) can never leak into
# the cost-math view as non-float values.
RATE_KEYS: tuple[str, ...] = ("input", "output", "cache_write", "cache_read")

# Cost-math view of the registry: model id → {input, output, cache_write,
# cache_read}. Derived (not restated) so it can never drift from
# MODEL_PRICING.
RATES: dict[str, dict[str, float]] = {
    model_id: {k: row[k] for k in RATE_KEYS}
    for model_id, row in MODEL_PRICING.items()
}

# Capability vocabulary. Open by design — new values are added here as the
# catalog grows (`"vision"`, `"transcription"`, `"tts"`, …); consumers should
# treat an unrecognized value as "not one I handle" rather than an error.
CHAT = "chat"              # text in, text out; tool-use capable
EMBEDDING = "embedding"    # text → vector
RERANKER = "reranker"      # (query, docs) → ordered relevance scores

CAPABILITIES: tuple[str, ...] = (CHAT, EMBEDDING, RERANKER)


class ModelRecord(BaseModel):
    """One model's static facts: identity, capability, and its rates.

    The typed view of a `MODEL_PRICING` row, with the model id folded in
    as `id`. Built by `catalog()`; `MODEL_PRICING` remains the source of
    truth and the dict form stays available for callers that already
    project it themselves.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    capability: str
    input: float
    output: float
    cache_write: float
    cache_read: float


def catalog(capability: str | None = None) -> list[ModelRecord]:
    """Every model in the registry, optionally filtered by capability.

    Args:
        capability: When given, return only models with this exact
            `capability` value (e.g. `pricing.CHAT`). Unrecognized
            values return an empty list rather than raising — callers
            filtering on a capability this version doesn't know about
            should see "no models", not a crash.

    Returns:
        `ModelRecord` list in registry order.

    Example:
        >>> [m.id for m in catalog(CHAT)][:2]
        ['claude-fable-5', 'claude-opus-5']
    """
    records = [
        ModelRecord(id=model_id, **row)
        for model_id, row in MODEL_PRICING.items()
    ]
    if capability is None:
        return records
    return [m for m in records if m.capability == capability]


class UnpricedModel(LookupError):
    """Raised when a model has no rate row and the caller refuses to proceed.

    `_cost` never raises this — post-flight pricing stays tolerant by
    design, because the money is already spent and an accounting failure
    must not break a response that succeeded. It is raised pre-flight by
    `budget.RequirePricedModelHook`, where aborting is still free.
    """


def is_priced(model: str) -> bool:
    """True when `model` has a rate row, and can therefore be costed.

    The predicate behind pre-flight enforcement. Note this answers only
    "can the token math run", not "is the resulting figure the complete
    bill" — a priced model can still carry non-token line items the
    library does not yet meter (server-side tool use; see issue #11).
    """
    return model in RATES


# Model ids already flagged this process as having no rate row — gates the
# warn-once alert/log in `_warn_unpriced`.
_unpriced_warned: set[str] = set()


def _warn_unpriced(model: str) -> None:
    """Warn once per process that `model` has no row in `MODEL_PRICING`.

    Fires an operator alert through the library alert seam and emits a
    log line the first time an unpriced model is costed, then stays
    quiet for that model id. The alert seam is best-effort — a delivery
    failure never breaks cost math.
    """
    if model in _unpriced_warned:
        return
    _unpriced_warned.add(model)
    _log.warning(
        "pricing: model %r has no entry in MODEL_PRICING; its spend is "
        "billed at $0 and undercounted until a rate row is added.", model,
    )
    # Deferred import to keep the cost-math hot path free of the alerts
    # import graph until the first unpriced model is seen.
    from llm_cost_governor.alerts import WARNING, alert

    alert(
        WARNING,
        "Unpriced model billed at $0",
        f"Model '{model}' has no entry in pricing.MODEL_PRICING, so its "
        f"spend is counted as $0 and undercounted against the demo caps. "
        f"Add a rate row to restore accurate accounting.",
    )


def _cost(model: str, input_tok: int, output_tok: int,
          cache_read_tok: int = 0, cache_write_tok: int = 0) -> float:
    """Compute the USD cost of one model call from token counts.

    Args:
        model: Model id; must match a key in `RATES`.
        input_tok: Input tokens billed at the model's `input` rate.
        output_tok: Output tokens billed at the model's `output` rate.
        cache_read_tok: Cache-hit input tokens billed at the
            `cache_read` rate (≈ 10% of full input).
        cache_write_tok: Cache-creation tokens billed at the
            `cache_write` rate (≈ 125% of full input).

    Returns:
        Cost in USD. Returns `0.0` for unknown models — defensive so that
        pricing-table lag never crashes a session. The first time an
        unpriced (non-empty) model id is costed, `_warn_unpriced` fires a
        one-time operator alert + log line.
    """
    r = RATES.get(model)
    if r is None:
        if model:  # skip the empty-string default (a usage dict missing `model`)
            _warn_unpriced(model)
        return 0.0
    M = 1_000_000
    return (
        input_tok      / M * r["input"]
        + output_tok   / M * r["output"]
        + cache_read_tok  / M * r["cache_read"]
        + cache_write_tok / M * r["cache_write"]
    )


def usd_for_usage(usage: dict) -> float:
    """USD cost of one completed call from an Anthropic ``usage``-shaped dict.

    Public convenience wrapper around `_cost` that maps the Anthropic
    response-usage field names (`input_tokens`, `cache_read_input_tokens`,
    `cache_creation_input_tokens`, …) to the calculator's arguments.

    Args:
        usage: Dict with keys `model`, `input_tokens`, `output_tokens`,
            `cache_read_input_tokens`, `cache_creation_input_tokens`.
            Missing keys default to 0 / unknown-model (→ $0, see `_cost`).

    Returns:
        Cost in USD.
    """
    # `or 0` coerces both missing keys and explicit `None` values (from
    # providers that don't expose a cache split, e.g. OpenAI / Voyage) —
    # None survives .get()'s default only for the missing-key case, so
    # both branches need to collapse to 0 for the arithmetic in _cost.
    return _cost(
        usage.get("model", ""),
        input_tok       = usage.get("input_tokens") or 0,
        output_tok      = usage.get("output_tokens") or 0,
        cache_read_tok  = usage.get("cache_read_input_tokens") or 0,
        cache_write_tok = usage.get("cache_creation_input_tokens") or 0,
    )
