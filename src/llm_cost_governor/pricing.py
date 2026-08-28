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

Rates as of 2026-08-01 (Claude, Voyage) and 2026-08-27 (OpenAI):
    Claude:  https://www.anthropic.com/pricing
    Voyage:  https://docs.voyageai.com/docs/pricing
    OpenAI:  https://developers.openai.com/api/docs/pricing
All figures in USD per 1 million tokens.

Two different reasons a row carries zeros, worth keeping distinct:

* Voyage embedding and rerank models have no output / cache tokens at
  all — the dimension does not exist for them, so ``output``,
  ``cache_write``, and ``cache_read`` are 0.0 and the existing ``_cost``
  arithmetic works unchanged (input tokens × input rate + zeros).
* OpenAI rows set the cache rates to 0.0 because this table does not
  model OpenAI prompt caching *yet*, not because the dimension is
  absent. That is correct only for uncached calls. Price a cached
  OpenAI call today and its cache tokens cost $0 and undercount — the
  same silent-undercount shape as issue #5. Enabling OpenAI caching
  means filling those rates in first.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

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

    # ── OpenAI GPT-5 family (current generation) ──
    # Cache rates are 0.00 throughout: OpenAI *does* have prompt caching
    # (unlike Voyage, which has no such dimension), but this table does not
    # model it yet. That is safe only while callers make uncached calls —
    # the moment a cached OpenAI call is priced, its cache tokens bill at
    # $0 and undercount. `test_openai_rows_have_no_cache_rates` pins the
    # assumption so enabling caching has to be a deliberate change here.
    "gpt-5.6-sol": {
        "label": "GPT-5.6 Sol",
        "capability": "chat",
        "input":        4.00,
        "output":      20.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.6-terra": {
        "label": "GPT-5.6 Terra",
        "capability": "chat",
        "input":        2.00,
        "output":      12.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.6-luna": {
        "label": "GPT-5.6 Luna",
        "capability": "chat",
        "input":        0.20,
        "output":       1.20,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.5": {
        "label": "GPT-5.5",
        "capability": "chat",
        "input":        5.00,
        "output":      30.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.5-pro": {
        "label": "GPT-5.5 Pro",
        "capability": "chat",
        "input":       30.00,
        "output":     180.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.4": {
        "label": "GPT-5.4",
        "capability": "chat",
        "input":        2.50,
        "output":      15.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.4-mini": {
        "label": "GPT-5.4 Mini",
        "capability": "chat",
        "input":        0.75,
        "output":       4.50,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.4-nano": {
        "label": "GPT-5.4 Nano",
        "capability": "chat",
        "input":        0.20,
        "output":       1.25,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.4-pro": {
        "label": "GPT-5.4 Pro",
        "capability": "chat",
        "input":       30.00,
        "output":     180.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.2": {
        "label": "GPT-5.2",
        "capability": "chat",
        "input":        1.75,
        "output":      14.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.2-pro": {
        "label": "GPT-5.2 Pro",
        "capability": "chat",
        "input":       21.00,
        "output":     168.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5.1": {
        "label": "GPT-5.1",
        "capability": "chat",
        "input":        1.25,
        "output":      10.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5": {
        "label": "GPT-5",
        "capability": "chat",
        "input":        1.25,
        "output":      10.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5-mini": {
        "label": "GPT-5 Mini",
        "capability": "chat",
        "input":        0.25,
        "output":       2.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5-nano": {
        "label": "GPT-5 Nano",
        "capability": "chat",
        "input":        0.05,
        "output":       0.40,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "gpt-5-pro": {
        "label": "GPT-5 Pro",
        "capability": "chat",
        "input":       15.00,
        "output":     120.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },

    # ── OpenAI embeddings (current generation) ──
    # Embeddings have no output or cache dimension, like the Voyage rows.
    "text-embedding-3-small": {
        "label": "OpenAI Embedding 3 Small",
        "capability": "embedding",
        "input":        0.02,
        "output":      0.00,
        "cache_write": 0.00,
        "cache_read":  0.00,
    },
    "text-embedding-3-large": {
        "label": "OpenAI Embedding 3 Large",
        "capability": "embedding",
        "input":        0.13,
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


# ── Server-side tool pricing ───────────────────────────────────────────────────
# Anthropic bills some server-side tool use *in addition to* tokens, and
# reports it as a sibling of the token counts on the same call:
#
#     "usage": {"input_tokens": 105, "output_tokens": 6039,
#               "server_tool_use": {"web_search_requests": 1}}
#
# Rates as of 2026-08-28 — https://platform.claude.com/docs/en/about-claude/pricing
#
# Keys map to USD *per request*. A key absent from this table is not priced;
# `_warn_unpriced_tool` fires once for it rather than letting it cost $0
# silently, which is the failure this whole seam exists to avoid.
SERVER_TOOL_PRICING: dict[str, float] = {
    # Web search: $10 per 1,000 searches. Each search counts as one use
    # regardless of how many results come back; failed searches aren't billed.
    "web_search_requests": 0.010,
    # Web fetch: no additional charge — you pay only for the fetched content
    # as input tokens, which the token math already covers. Priced at 0.0
    # explicitly so it reads as "known to be free", not "forgotten".
    "web_fetch_requests": 0.0,
    # NOT here, deliberately: `code_execution_requests`. Code execution is
    # billed by container-hour ($0.05/hour, 1,550 free hours/month, 5-minute
    # minimum), not per request — a per-request rate would be fiction. It
    # warns instead, so the gap is visible rather than silently $0.
}

# Server-tool keys already flagged this process as unpriced.
_unpriced_tools_warned: set[str] = set()


def _warn_unpriced_tool(key: str) -> None:
    """Warn once per process that server-tool `key` has no rate.

    Mirrors `_warn_unpriced` for the non-token billing dimension: the
    library would otherwise count a real, billed line item as $0 with no
    signal at all.
    """
    if key in _unpriced_tools_warned:
        return
    _unpriced_tools_warned.add(key)
    _log.warning(
        "pricing: server tool %r has no entry in SERVER_TOOL_PRICING; its "
        "spend is counted as $0 and undercounted against budgets and caps.",
        key,
    )
    from llm_cost_governor.alerts import WARNING, alert

    alert(
        WARNING,
        "Unpriced server tool billed at $0",
        f"Server tool {key!r} has no entry in pricing.SERVER_TOOL_PRICING, so "
        f"its spend is counted as $0. Add a rate, or price it out of band.",
    )


def server_tool_cost(server_tool_use: Mapping[str, int] | None) -> float:
    """USD for the server-side tool use reported on one call.

    Args:
        server_tool_use: The provider's ``usage.server_tool_use`` mapping
            (request counts by tool key), or None when the call made no
            server-tool use.

    Returns:
        Cost in USD, on top of whatever the token math returns. Unpriced
        keys contribute 0.0 and fire `_warn_unpriced_tool` once.
    """
    if not server_tool_use:
        return 0.0
    total = 0.0
    for key, count in server_tool_use.items():
        rate = SERVER_TOOL_PRICING.get(key)
        if rate is None:
            _warn_unpriced_tool(key)
            continue
        total += (count or 0) * rate
    return total


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
        Cost in USD — token cost plus any server-side tool use reported
        under ``server_tool_use``. Folding the two together here is what
        makes non-token spend visible to every budget, cap, and total
        downstream, all of which price through this one function.
    """
    # `or 0` coerces both missing keys and explicit `None` values (from
    # providers that don't expose a cache split, e.g. OpenAI / Voyage) —
    # None survives .get()'s default only for the missing-key case, so
    # both branches need to collapse to 0 for the arithmetic in _cost.
    return server_tool_cost(usage.get("server_tool_use")) + _cost(
        usage.get("model", ""),
        input_tok       = usage.get("input_tokens") or 0,
        output_tok      = usage.get("output_tokens") or 0,
        cache_read_tok  = usage.get("cache_read_input_tokens") or 0,
        cache_write_tok = usage.get("cache_creation_input_tokens") or 0,
    )
