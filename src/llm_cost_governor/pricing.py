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

All figures in USD per 1 million tokens. When each vendor's rates were
last checked is recorded in `RATES_AS_OF` — machine-readable, because a
prose date nobody reads is how a table goes quietly wrong.

Two different reasons a row carries zeros, worth keeping distinct:

* Voyage embedding and rerank models have no output / cache tokens at
  all — the dimension does not exist for them, so ``output``,
  ``cache_write``, and ``cache_read`` are 0.0 and the existing ``_cost``
  arithmetic works unchanged (input tokens × input rate + zeros).
* OpenAI rows carry real cache rates as of 0.4.3, now that the adapter
  reports cached tokens. ``cache_read`` is OpenAI's discounted
  cached-input rate (10% of input on models that support caching).
  ``cache_write`` is 0.0 on most models because OpenAI's automatic
  caching charges nothing to populate the cache — that zero means
  "free", not "unmodelled". The three ``gpt-5.6-*`` models are the
  exception and do list a cache-write fee.
* The four ``-pro`` models list no cached-input rate at all: they do not
  support prompt caching, so both cache fields are 0.0.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_log = logging.getLogger(__name__)

class InvalidRateRow(ValueError):
    """Raised when a rate row is malformed or incomplete, from the shipped
    table at import or from `register_rates` at runtime."""


# ── Model registry (single source of truth) ───────────────────────────────────
# The table lives in `pricing.json`, beside this module and shipped in the
# wheel. It is data, so it is read rather than executed — a malformed row
# fails at import with a validation error instead of at first call.
#
# Each row carries `label`, `capability`, `provider`, an optional `note`, and
# EITHER four flat rates (`input` / `output` / `cache_write` / `cache_read`)
# OR a `tiers` list for a model whose rate depends on prompt size.
#
# `note` exists because JSON has no comments and the reasoning behind some
# values is load-bearing — it is what stops a future reader "correcting" a
# deliberately non-standard rate.
#
# Keys are BARE model aliases (`claude-haiku-4-5`), never dated snapshots: a
# dated key bills the floating alias callers actually use at $0 (issue #5).
# Enforced by `test_model_pricing_keys_are_bare_aliases`.

# The four USD-per-1M-token rate keys. Declared here because the loader
# validates against them before anything else in the module runs.
RATE_KEYS: tuple[str, ...] = ("input", "output", "cache_write", "cache_read")

_TABLE_PATH = Path(__file__).with_name("pricing.json")


def _load_table(path: Path) -> dict[str, dict]:
    """Read and validate the shipped pricing table.

    Raises:
        InvalidRateRow: On a row missing required fields, carrying both flat
            rates and tiers, or with a malformed tier list. Failing at import
            is deliberate — a half-valid pricing table is a silently wrong
            bill.
    """
    with path.open(encoding="utf-8") as fh:
        table = json.load(fh)
    for model, row in table.items():
        missing = {"label", "capability", "provider"} - set(row)
        if missing:
            raise InvalidRateRow(f"{model!r} missing {sorted(missing)} in {path.name}")
        has_flat = any(k in row for k in RATE_KEYS)
        if ("tiers" in row) == has_flat:
            raise InvalidRateRow(
                f"{model!r} must carry either flat rates or `tiers`, not both/neither"
            )
        if "tiers" in row:
            _validate_tiers(model, row["tiers"])
        else:
            absent = set(RATE_KEYS) - set(row)
            if absent:
                raise InvalidRateRow(f"{model!r} missing rate keys {sorted(absent)}")
    return table


def _validate_tiers(model: str, tiers: object) -> None:
    """Check a tier list: ascending thresholds, exactly one open-ended tier last."""
    if not isinstance(tiers, list) or len(tiers) < 2:
        raise InvalidRateRow(f"{model!r}: `tiers` must be a list of at least two tiers")
    seen_open = False
    last_threshold = 0
    for i, tier in enumerate(tiers):
        absent = set(RATE_KEYS) - set(tier)
        if absent:
            raise InvalidRateRow(f"{model!r} tier {i} missing {sorted(absent)}")
        cap = tier.get("max_input_tokens")
        if cap is None:
            if i != len(tiers) - 1:
                raise InvalidRateRow(
                    f"{model!r}: only the last tier may omit `max_input_tokens`; "
                    f"tier {i} does, leaving later tiers unreachable"
                )
            seen_open = True
        else:
            if cap <= last_threshold:
                raise InvalidRateRow(
                    f"{model!r} tier {i}: `max_input_tokens` must ascend "
                    f"({cap} follows {last_threshold})"
                )
            last_threshold = cap
    if not seen_open:
        raise InvalidRateRow(
            f"{model!r}: the last tier must omit `max_input_tokens` so every "
            f"prompt size resolves to a rate"
        )


MODEL_PRICING: dict[str, dict] = _load_table(_TABLE_PATH)

# The four USD-per-1M-token rate keys. `RATES` projects exactly these — an
# allowlist, not "everything except `label`", so descriptive fields added to
# the registry (`capability`, and whatever comes next) can never leak into
# the cost-math view as non-float values.

# Cost-math view of the registry: model id → {input, output, cache_write,
# cache_read}. Derived (not restated) so it can never drift from
# MODEL_PRICING.
# Cost-math view of the registry, for FLAT-RATE models only. A tiered model
# has no single rate, so inventing one here would be a number that is correct
# on one side of its threshold and wrong on the other — exactly the silent
# mispricing this module exists to avoid. Tiered models are therefore absent
# from `RATES` and resolved by `_cost` instead; use `is_priced()`, not
# `model in RATES`, to ask whether a model can be costed.
RATES: dict[str, dict[str, float]] = {
    model_id: {k: row[k] for k in RATE_KEYS}
    for model_id, row in MODEL_PRICING.items()
    if "tiers" not in row
}

# Immutable snapshot of the table as shipped, taken before any caller override
# can touch it. `reset_overrides()` restores from this rather than re-deriving,
# so a test that overrides a shipped model puts back the exact verified row.
_SHIPPED_PRICING: dict[str, dict] = {m: dict(r) for m, r in MODEL_PRICING.items()}

# ── Rate provenance ───────────────────────────────────────────────────────────
# When each vendor's published rates were last checked against the source
# below. Machine-readable on purpose: a stale rate is the one failure the
# pre-flight gate cannot see — `is_priced()` returns True for a row whose
# numbers are wrong, so the call is admitted and every budget downstream is
# computed from a bad figure, silently. `test_rate_provenance_is_fresh`
# turns that into a build failure on a deadline instead.
#
# Updating a rate means updating its date here in the same commit.
RATE_SOURCES: dict[str, str] = {
    "anthropic": "https://www.anthropic.com/pricing",
    "openai": "https://developers.openai.com/api/docs/pricing",
    "voyage": "https://docs.voyageai.com/docs/pricing",
    # Server-side tool rates (web search, web fetch) — see SERVER_TOOL_PRICING.
    "anthropic-server-tools": "https://platform.claude.com/docs/en/about-claude/pricing",
}

RATES_AS_OF: dict[str, date] = {
    "anthropic": date(2026, 9, 29),
    "openai": date(2026, 9, 29),
    "voyage": date(2026, 8, 1),
    "anthropic-server-tools": date(2026, 8, 28),
}

# Days after which a vendor's rates are considered worth re-checking (warn)
# and stale enough to fail the build (fail). The gap between them is the
# window to act in before an unrelated PR goes red.
RATES_STALE_WARN_DAYS = 90
RATES_STALE_FAIL_DAYS = 180


def rates_age_days(source: str, *, today: date | None = None) -> int:
    """Days since `source`'s rates were last checked against its vendor page.

    Args:
        source: A key of `RATES_AS_OF` (a provider name, or
            ``"anthropic-server-tools"`` for the non-token rates).
        today: Overrides the current date; for tests. Defaults to
            today in UTC — rate freshness is measured in months, so the
            zone only matters for reproducibility.

    Returns:
        Age in days.

    Raises:
        KeyError: If `source` has no recorded provenance.
    """
    return ((today or datetime.now(tz=UTC).date()) - RATES_AS_OF[source]).days


def stalest_rates(*, today: date | None = None) -> tuple[str, int]:
    """The source whose rates were checked longest ago, and its age in days.

    For apps that want to surface pricing freshness alongside spend — the
    library has no way to tell you a rate is wrong, only how long it has
    been since anyone looked.
    """
    ages = {src: rates_age_days(src, today=today) for src in RATES_AS_OF}
    worst = max(ages, key=lambda k: ages[k])
    return worst, ages[worst]


def load_rates(source: object, *, object_name: str = "rates.json") -> None:
    """Register rate overrides from a JSON (or YAML) document.

    The file form of `register_rates`, so a price correction is a config
    change rather than a code change. Accepts:

    * a path (`str` / `os.PathLike`) — read from disk;
    * a `StateBackend` — read `object_name` through it, which is how a rates
      file can live in the same GCS bucket as your counters and be edited
      without rebuilding an image;
    * a `str` of JSON — if you fetched it yourself.

    **This library never reads an environment variable** (see
    `docs/integration.md`), so where the path comes from is your config
    layer's business, not ours.

    JSON always works — it is stdlib, and the core install deliberately
    carries one dependency. `.yaml` / `.yml` paths additionally need
    `pyyaml`, available as the ``[yaml]`` extra; without it you get an
    ImportError naming the extra rather than a parse failure.

    The document is one object of ``{model_id: row}``, exactly the shape
    `register_rates` takes::

        {"claude-sonnet-5": {"input": 2.00, "output": 10.00}}

    Raises:
        InvalidRateRow: On a malformed document or row.
    """
    import json
    import os

    text: str
    if hasattr(source, "read") and callable(source.read):     # StateBackend
        blob = source.read(object_name)
        if blob is None:
            raise InvalidRateRow(f"no rates object {object_name!r} in that backend")
        text, name = blob, object_name
    elif isinstance(source, str) and source.lstrip()[:1] in "{[":
        # A document, not a path. Both JSON braces are accepted so that a
        # non-object document fails with "expected an object" rather than a
        # confusing FileNotFoundError about its own first line.
        text, name = source, "<string>"
    else:
        path = os.fspath(source)  # type: ignore[arg-type]
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        name = path

    if name.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:                            # pragma: no cover
            raise ImportError(
                f"parsing {name} needs PyYAML — install llm-cost-governor[yaml], "
                f"or use JSON, which needs no extra"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise InvalidRateRow(
            f"{name}: expected an object of {{model_id: row}}, got {type(data).__name__}"
        )
    register_rates(data)


# Provider vocabulary. These strings are also the names
# `providers.get_provider` accepts for the subset that has adapters, so a
# caller can test `m.provider in providers.ADAPTERS` to find out whether
# `guarded_call` can wrap the model or whether it must use `record_usage`.
PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "voyage")

# Capability vocabulary. Open by design — new values are added here as the
# catalog grows (`"vision"`, `"transcription"`, `"tts"`, …); consumers should
# treat an unrecognized value as "not one I handle" rather than an error.
CHAT = "chat"              # text in, text out; tool-use capable
EMBEDDING = "embedding"    # text → vector
RERANKER = "reranker"      # (query, docs) → ordered relevance scores

CAPABILITIES: tuple[str, ...] = (CHAT, EMBEDDING, RERANKER)


class ModelRecord(BaseModel):
    """One model's static facts: identity, provider, capability, rates.

    The typed view of a `MODEL_PRICING` row, with the model id folded in
    as `id`. Built by `catalog()`; `MODEL_PRICING` remains the source of
    truth and the dict form stays available for callers that already
    project it themselves.

    ``provider`` names the vendor, using the same strings as
    `providers.get_provider` — so ``m.provider in providers.ADAPTERS``
    answers "can `guarded_call` wrap this model?" directly. It exists so
    consumers stop inferring the vendor from id prefixes: that heuristic
    is right until it silently isn't, on the first family whose ids match
    none of the patterns a caller happened to write.

    Note the deliberate asymmetry with `ADAPTERS`: every row has a
    ``provider``, but not every provider has an adapter. `voyage` rows
    are priced and metered through `record_usage` with no adapter, by
    design.

    **Tiered models.** When ``tiered`` is True the model's rate depends on
    prompt size, and the four rate fields report the **entry tier** — the
    cheapest one, matching how vendors present these ("From $0.10 / MTok").
    ``tiers`` carries the full schedule. Those fields are for display; any
    actual cost must come from `_cost` / `usd_for_usage`, which select the
    right tier from the call's token counts. Reading ``input`` to compute a
    bill would undercount every call above the threshold.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    capability: str
    provider: str
    input: float
    output: float
    cache_write: float
    cache_read: float
    tiered: bool = False
    tiers: list[dict] | None = None
    note: str | None = None


def _record_for(model_id: str, row: Mapping[str, object]) -> ModelRecord:
    """Build a `ModelRecord`, flattening a tiered row to its entry tier."""
    if "tiers" in row:
        entry = row["tiers"][0]                      # type: ignore[index]
        rates = {k: float(entry[k]) for k in RATE_KEYS}
        return ModelRecord(
            id=model_id, label=row["label"], capability=row["capability"],   # type: ignore[arg-type]
            provider=row["provider"], tiered=True,                            # type: ignore[arg-type]
            tiers=list(row["tiers"]), note=row.get("note"), **rates,          # type: ignore[arg-type]
        )
    return ModelRecord(id=model_id, **row)                                    # type: ignore[arg-type]


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
    records = [_record_for(model_id, row) for model_id, row in MODEL_PRICING.items()]
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


# ── Caller-supplied rate overrides ────────────────────────────────────────────
# Model ids whose rates came from the caller rather than this table. Kept so
# overrides stay visible: the library cannot vouch for a rate it did not
# verify, and silent pricing is the failure this whole module exists to avoid.
_overridden: set[str] = set()


def _validate_row(model: str, row: Mapping[str, object], *, is_new: bool) -> dict:
    """Check one registered row and return it as a plain dict.

    A new model needs a complete record — `catalog()` filters on `capability`
    and `provider`, so a row missing them would price correctly and then be
    invisible to any consumer that lists models. An existing model needs only
    the rates being changed.
    """
    out = dict(row)
    if is_new:
        missing = {"label", "capability", "provider", *RATE_KEYS} - set(out)
        if missing:
            raise InvalidRateRow(
                f"{model!r} is not in MODEL_PRICING, so registering it needs a "
                f"complete record; missing: {sorted(missing)}"
            )
    unknown_rate_keys = {k for k in out if k in RATE_KEYS}
    for k in unknown_rate_keys:
        v = out[k]
        if not isinstance(v, int | float) or isinstance(v, bool):
            raise InvalidRateRow(f"{model!r}.{k} must be a number, got {v!r}")
        if v < 0:
            raise InvalidRateRow(f"{model!r}.{k} must not be negative, got {v}")
    if is_new and out.get("input", 0) <= 0:
        raise InvalidRateRow(f"{model!r} has a non-positive input rate")
    return out


def register_rates(rows: Mapping[str, Mapping[str, object]]) -> None:
    """Overlay caller-supplied rates on top of the shipped pricing table.

    This is how an application prices a model the library does not carry, or
    corrects one it carries wrongly, **without waiting for a release here**.

    Semantics are an overlay, per model:

    * An **existing** model id takes a partial update — give only the rates
      you are changing; `label` / `capability` / `provider` are kept.
    * A **new** model id needs a complete record, because `catalog()` filters
      on `capability` and `provider`.
    * Nothing is ever removed, so upgrading still brings you our corrections —
      but an override is **sticky**: yours keeps winning until you drop it.
      Treat one as a temporary patch, not a fix.

    Registration updates `MODEL_PRICING` and `RATES` together, so a registered
    model is immediately priced by `_cost`, admitted by `is_priced` and
    therefore by `RequirePricedModelHook`, and listed by `catalog()`. Pricing a
    model that the pre-flight gate then refuses would be worse than the gap
    this closes, so the three are kept consistent by construction.

    Call once at startup, before serving traffic. Registration is not
    synchronised; it is a startup action, not a runtime one.

    Args:
        rows: Model id → row. A row may carry any of `RATE_KEYS` plus
            `label` / `capability` / `provider`.

    Raises:
        InvalidRateRow: On a malformed row, or an incomplete one for a model
            not already in the table. Nothing is applied if any row is bad —
            a half-applied override is a silently wrong price.

    Example:
        >>> register_rates({"claude-sonnet-5": {"input": 2.00, "output": 10.00}})
        >>> register_rates({"acme-1": {"label": "Acme 1", "capability": "chat",
        ...                            "provider": "acme", "input": 1.0,
        ...                            "output": 3.0, "cache_write": 0.0,
        ...                            "cache_read": 0.0}})
    """
    # Validate everything first: applying some rows and rejecting others would
    # leave the table in a state nobody wrote down.
    validated = {
        model: _validate_row(model, row, is_new=model not in MODEL_PRICING)
        for model, row in rows.items()
    }
    for model, row in validated.items():
        merged = {**MODEL_PRICING.get(model, {}), **row}
        MODEL_PRICING[model] = merged
        RATES[model] = {k: float(merged[k]) for k in RATE_KEYS}
        _overridden.add(model)
        _log.info(
            "pricing: rates for %r are caller-supplied, not from this table's "
            "verified figures", model,
        )


def overridden_models() -> frozenset[str]:
    """Model ids whose rates were supplied by the caller via `register_rates`.

    Exposed so an app can surface "N rates are locally overridden" beside its
    spend — the freshness machinery deliberately ignores these, because the
    library has no basis to call a caller-supplied rate fresh or stale.
    """
    return frozenset(_overridden)


def reset_overrides() -> None:
    """Drop every registered override, restoring the shipped table. For tests."""
    for model in list(_overridden):
        shipped = _SHIPPED_PRICING.get(model)
        if shipped is None:
            MODEL_PRICING.pop(model, None)
            RATES.pop(model, None)
        else:
            MODEL_PRICING[model] = dict(shipped)
            if "tiers" in shipped:
                RATES.pop(model, None)   # tiered models are absent from RATES
            else:
                RATES[model] = {k: float(shipped[k]) for k in RATE_KEYS}
    _overridden.clear()


def is_priced(model: str) -> bool:
    """True when `model` has a rate row, and can therefore be costed.

    The predicate behind pre-flight enforcement. Reads `MODEL_PRICING`
    rather than `RATES`, because a tiered model is absent from the latter
    by design and is still perfectly costable.

    Note this answers only "can the token math run", not "is the resulting
    figure the complete bill" — a priced model can still carry non-token
    line items the library does not yet meter (server-side tool use; see
    issue #11).
    """
    return model in MODEL_PRICING


# ── Server-side tool pricing ───────────────────────────────────────────────────
# Anthropic bills some server-side tool use *in addition to* tokens, and
# reports it as a sibling of the token counts on the same call:
#
#     "usage": {"input_tokens": 105, "output_tokens": 6039,
#               "server_tool_use": {"web_search_requests": 1}}
#
# Rates and source: RATE_SOURCES / RATES_AS_OF['anthropic-server-tools'].
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


def _rates_for(model: str, input_tok: int) -> dict[str, float] | None:
    """The rate row to bill `model` at, given this call's input size.

    Flat models return their single row. A tiered model selects the first
    tier whose `max_input_tokens` the prompt does not exceed, falling through
    to the open-ended last tier — which the loader guarantees exists, so every
    prompt size resolves. Returns None for a model with no rates at all.

    Selection is on **input tokens**, matching how vendors describe these
    tiers ("for prompts up to N tokens"). Cached and cache-write tokens are
    billed at the selected tier's rates but do not themselves move the
    threshold; `input_tok` as passed is what decides.
    """
    flat = RATES.get(model)
    if flat is not None:
        return flat
    row = MODEL_PRICING.get(model)
    if row is None or "tiers" not in row:
        return None
    for tier in row["tiers"]:
        cap = tier.get("max_input_tokens")
        if cap is None or input_tok <= cap:
            return {k: float(tier[k]) for k in RATE_KEYS}
    return None  # pragma: no cover — loader guarantees an open-ended last tier


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
    r = _rates_for(model, input_tok)
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
