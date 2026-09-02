# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Unit tests for pricing.py: cost math and ScopeBudget enforcement."""

import pytest

from llm_cost_governor.budget import BudgetExceeded, ScopeBudget
from llm_cost_governor.pricing import RATES, _cost

# ── _cost() ────────────────────────────────────────────────────────────────────

def test_cost_sonnet_baseline():
    # Sonnet: $3/M input, $15/M output.
    assert _cost("claude-sonnet-4-6", input_tok=1_000_000, output_tok=0) == pytest.approx(3.00)
    assert _cost("claude-sonnet-4-6", input_tok=0, output_tok=1_000_000) == pytest.approx(15.00)


def test_cost_opus_47_baseline():
    # Opus 4.7: $5/M input, $25/M output.
    assert _cost("claude-opus-4-7", input_tok=1_000_000, output_tok=1_000_000) == pytest.approx(30.00)


def test_cost_haiku_baseline():
    # Haiku 4.5: $1/M input, $5/M output (was wrongly carrying retired
    # Haiku 3.5 rates of $0.80/$4.00 before #261).
    assert _cost("claude-haiku-4-5", input_tok=1_000_000, output_tok=1_000_000) == pytest.approx(6.00)


def test_cost_includes_cache_read_and_write():
    # Sonnet rates: cache_write=$3.75/M, cache_read=$0.30/M.
    cost = _cost(
        "claude-sonnet-4-6",
        input_tok=0,
        output_tok=0,
        cache_read_tok=1_000_000,
        cache_write_tok=1_000_000,
    )
    assert cost == pytest.approx(0.30 + 3.75)


def test_cost_unknown_model_returns_zero():
    # Defensive: unknown models cost zero rather than raising — keeps the UI
    # alive when a new model id slips into a session log.
    assert _cost("not-a-real-model", input_tok=999_999, output_tok=999_999) == 0.0


def test_rate_table_covers_models_in_config():
    # Smoke: every key in RATES is a non-empty dict with the four required rates.
    # `input` must be positive (every model bills for input); output / cache
    # rates may be 0 for providers without those token dimensions — Voyage
    # embeddings, for example, have no output or cache concept.
    required = {"input", "output", "cache_read", "cache_write"}
    for model, rates in RATES.items():
        assert required.issubset(rates.keys()), f"{model} missing rate keys"
        assert rates["input"] > 0, f"{model} has non-positive input rate"
        for k in ("output", "cache_read", "cache_write"):
            assert rates[k] >= 0, f"{model} has negative {k} rate"


# ── MODEL_PRICING registry / config.MODELS projection (#261) ────────────────────

def test_model_pricing_rows_are_complete():
    # Every registry row carries a label plus the four rates. Only `input`
    # must be positive — output / cache rates can be 0 for providers that
    # don't bill those token dimensions (Voyage embeddings + rerank).
    from llm_cost_governor.pricing import MODEL_PRICING

    required = {"label", "input", "output", "cache_read", "cache_write"}
    for model, row in MODEL_PRICING.items():
        assert required.issubset(row.keys()), f"{model} missing fields"
        assert isinstance(row["label"], str) and row["label"], f"{model} bad label"
        assert row["input"] > 0, f"{model} has non-positive input rate"
        for k in ("output", "cache_read", "cache_write"):
            assert row[k] >= 0, f"{model} has negative {k} rate"


def test_model_pricing_keys_are_bare_aliases():
    # Regression guard for #5. Keys must be bare aliases, never dated
    # snapshots: a caller naming `claude-haiku-4-5` (the documented way to
    # name a current model) got no rate match against a dated key, so its
    # spend was billed at $0 and undercounted against ScopeBudget ceilings.
    # The miss is quiet — a warn-once log line, not an error — so a budget
    # cap can be under-enforced without anyone noticing.
    import re

    from llm_cost_governor.pricing import MODEL_PRICING

    dated = [m for m in MODEL_PRICING if re.search(r"-20\d{6}$", m)]
    assert not dated, (
        f"dated snapshot keys in MODEL_PRICING: {dated}. Key the bare alias "
        f"instead — a dated key bills that model at $0 for callers using the "
        f"floating alias."
    )


def test_cost_resolves_bare_alias_for_every_model():
    # The table is the contract: every advertised key must actually price.
    # Guards against a row being present but unreachable through `_cost`.
    from llm_cost_governor.pricing import MODEL_PRICING

    for model in MODEL_PRICING:
        assert _cost(model, input_tok=1_000_000, output_tok=0) > 0, (
            f"{model} is in MODEL_PRICING but prices at $0"
        )


# ── capability / catalog() (#6) ─────────────────────────────────────────────────

def test_every_row_declares_a_known_capability():
    # The taxonomy lives upstream so consumers don't re-derive it from
    # provider-name prefixes. A row without a capability would silently
    # vanish from every filtered catalog() call.
    from llm_cost_governor.pricing import CAPABILITIES, MODEL_PRICING

    for model, row in MODEL_PRICING.items():
        assert "capability" in row, f"{model} has no capability"
        assert row["capability"] in CAPABILITIES, (
            f"{model} has unknown capability {row['capability']!r}; "
            f"add it to CAPABILITIES if it is a new one"
        )


def test_rates_projects_only_rate_keys():
    # Regression guard: RATES used to be "every key except `label`", so any
    # descriptive field added to the registry leaked into the cost-math view
    # as a non-float. It is now an explicit RATE_KEYS allowlist.
    from llm_cost_governor.pricing import RATE_KEYS, RATES

    for model, rates in RATES.items():
        assert set(rates) == set(RATE_KEYS), f"{model} rate keys drifted: {set(rates)}"
        for k, v in rates.items():
            assert isinstance(v, float), f"{model}.{k} is {type(v).__name__}, not float"


def test_catalog_covers_registry_and_filters_by_capability():
    from llm_cost_governor.pricing import (
        CAPABILITIES,
        CHAT,
        MODEL_PRICING,
        catalog,
    )

    assert len(catalog()) == len(MODEL_PRICING)
    assert [m.id for m in catalog()] == list(MODEL_PRICING)

    # Filters partition the catalog — every model lands in exactly one bucket.
    assert sum(len(catalog(c)) for c in CAPABILITIES) == len(MODEL_PRICING)
    assert all(m.capability == CHAT for m in catalog(CHAT))
    assert "claude-opus-5" in {m.id for m in catalog(CHAT)}


def test_catalog_unknown_capability_returns_empty_not_error():
    # A consumer filtering on a capability this version doesn't know about
    # should see "no models", not a crash — the vocabulary is open.
    from llm_cost_governor.pricing import catalog

    assert catalog("vision") == []


def test_catalog_records_carry_the_registry_rates():
    from llm_cost_governor.pricing import MODEL_PRICING, RATE_KEYS, catalog

    for m in catalog():
        row = MODEL_PRICING[m.id]
        assert m.label == row["label"]
        for k in RATE_KEYS:
            assert getattr(m, k) == row[k], f"{m.id}.{k} drifted from the registry"


def test_embedding_and_reranker_models_are_not_chat():
    # The motivating bug: Pitchcraft listed Voyage embedding + rerank models
    # as valid options for its chat stages, because it iterated the catalog
    # unfiltered. catalog(CHAT) must exclude them.
    from llm_cost_governor.pricing import CHAT, catalog

    chat_ids = {m.id for m in catalog(CHAT)}
    assert not any(i.startswith(("voyage-", "rerank-")) for i in chat_ids)


# ── OpenAI rows ────────────────────────────────────────────────────────────────

_OPENAI_PREFIXES = ("gpt-", "text-embedding-")


def _openai_ids():
    from llm_cost_governor.pricing import MODEL_PRICING

    return [m for m in MODEL_PRICING if m.startswith(_OPENAI_PREFIXES)]


def test_openai_rows_are_present_and_priced():
    from llm_cost_governor.pricing import _cost

    ids = _openai_ids()
    assert ids, "no OpenAI rows in the registry"
    for model in ids:
        assert _cost(model, input_tok=1_000_000, output_tok=0) > 0, f"{model} prices at $0"


def test_openai_cache_rates_are_a_discount_or_absent():
    # Superseded the 0.4.1 guard (which asserted these were all zero) once the
    # adapter began reporting cached tokens. The invariant now: a model either
    # supports caching, in which case cache_read is a real discount below the
    # input rate, or it does not, in which case both cache fields are zero.
    from llm_cost_governor.pricing import MODEL_PRICING

    for model in [m for m in _openai_ids() if m.startswith("gpt-")]:
        row = MODEL_PRICING[model]
        cr = row["cache_read"]
        assert cr == 0.0 or 0 < cr < row["input"], (
            f"{model} cache_read={cr} is neither absent nor a discount on "
            f"input={row['input']}"
        )
        assert row["cache_write"] >= 0.0


def test_openai_chat_and_embedding_models_are_tagged():
    from llm_cost_governor.pricing import CHAT, EMBEDDING, MODEL_PRICING

    for model in _openai_ids():
        cap = MODEL_PRICING[model]["capability"]
        expected = EMBEDDING if model.startswith("text-embedding-") else CHAT
        assert cap == expected, f"{model} tagged {cap!r}, expected {expected!r}"


def test_openai_models_are_reachable_through_the_priced_gate():
    # The pre-flight gate (#10) refuses unpriced models, so a missing row is
    # now a hard failure rather than a silent $0. Every OpenAI id we advertise
    # must pass it.
    from llm_cost_governor.pricing import is_priced

    for model in _openai_ids():
        assert is_priced(model), f"{model} would be refused by RequirePricedModelHook"


# ── provider on model records ──────────────────────────────────────────────────

def test_every_row_declares_a_known_provider():
    # A row without a provider would land as a validation error in catalog();
    # one with an unrecognized value would silently break a consumer routing
    # on it. Both fail here instead.
    from llm_cost_governor.pricing import MODEL_PRICING, PROVIDERS

    for model, row in MODEL_PRICING.items():
        assert "provider" in row, f"{model} has no provider"
        assert row["provider"] in PROVIDERS, (
            f"{model} has unknown provider {row['provider']!r}; add it to "
            f"PROVIDERS if it is a new vendor"
        )


def test_provider_values_match_get_provider_names():
    # The whole point of the field is that a consumer can hand it straight to
    # get_provider() rather than translating. Every provider that HAS an
    # adapter must therefore resolve under its own name.
    from llm_cost_governor.pricing import PROVIDERS
    from llm_cost_governor.providers import ADAPTERS, get_provider

    assert ADAPTERS <= set(PROVIDERS), (
        f"adapter names not in PROVIDERS: {ADAPTERS - set(PROVIDERS)}"
    )
    for name in ADAPTERS:
        assert get_provider(name) is not None


def test_not_every_provider_has_an_adapter():
    # Documents the deliberate asymmetry so it does not read as an oversight.
    # Voyage rows are priced and metered via record_usage with no adapter.
    from llm_cost_governor.pricing import catalog
    from llm_cost_governor.providers import ADAPTERS

    voyage = [m for m in catalog() if m.provider == "voyage"]
    assert voyage, "expected Voyage rows"
    assert "voyage" not in ADAPTERS


def test_provider_partitions_the_catalog():
    from llm_cost_governor.pricing import MODEL_PRICING, PROVIDERS, catalog

    assert sum(
        len([m for m in catalog() if m.provider == p]) for p in PROVIDERS
    ) == len(MODEL_PRICING)


def test_provider_is_not_inferable_from_id_prefix_alone():
    # The reason the field exists: id prefixes do not partition cleanly.
    # `text-embedding-3-small` is OpenAI but shares no prefix with `gpt-`,
    # and `rerank-2` is Voyage but shares none with `voyage-`.
    from llm_cost_governor.pricing import MODEL_PRICING

    assert MODEL_PRICING["text-embedding-3-small"]["provider"] == "openai"
    assert MODEL_PRICING["rerank-2"]["provider"] == "voyage"


def test_catalog_records_expose_provider():
    from llm_cost_governor.pricing import CHAT, catalog

    chat = {m.id: m.provider for m in catalog(CHAT)}
    assert chat["claude-opus-5"] == "anthropic"
    assert chat["gpt-5"] == "openai"


def test_cost_unknown_model_warns_operator_once(monkeypatch):
    """Unknown model still costs $0, but pings the registered AlertSink once
    (per process) instead of silently undercounting."""
    import llm_cost_governor.alerts as alerts_mod
    from llm_cost_governor import pricing

    calls: list[tuple[str, str, str]] = []

    class _CaptureSink:
        def alert(self, level, title, body):
            calls.append((level, title, body))

    monkeypatch.setattr(pricing, "_unpriced_warned", set())
    alerts_mod.set_alert_sink(_CaptureSink())

    model = "totally-made-up-model-261"
    assert pricing._cost(model, input_tok=1_000_000, output_tok=1_000_000) == 0.0
    assert pricing._cost(model, input_tok=1_000_000, output_tok=0) == 0.0  # second call

    # Costed twice, alerted once.
    assert len(calls) == 1
    assert calls[0][0] == alerts_mod.WARNING


def test_cost_empty_model_does_not_alert(monkeypatch):
    """A usage dict missing `model` (→ "") must not spam the AlertSink."""
    import llm_cost_governor.alerts as alerts_mod
    from llm_cost_governor import pricing

    calls: list[tuple[str, str, str]] = []

    class _CaptureSink:
        def alert(self, level, title, body):
            calls.append((level, title, body))

    monkeypatch.setattr(pricing, "_unpriced_warned", set())
    alerts_mod.set_alert_sink(_CaptureSink())

    assert pricing._cost("", input_tok=1_000, output_tok=1_000) == 0.0
    assert calls == []


# ── ScopeBudget ──────────────────────────────────────────────────────────────

def test_session_budget_initial_state():
    budget = ScopeBudget(limit_usd=1.00)
    assert budget.spent_usd == 0.0
    assert budget.limit_usd == 1.00


def test_would_exceed_false_when_within_limit():
    budget = ScopeBudget(limit_usd=1.00)
    # 1k input + 1k output on Sonnet ≈ $0.018 — well under $1.
    assert not budget.would_exceed(_cost("claude-sonnet-4-6", 1_000, 1_000))


def test_would_exceed_true_when_over_limit():
    budget = ScopeBudget(limit_usd=0.01)
    # 1M output on Opus = $25 — far over $0.01.
    assert budget.would_exceed(_cost("claude-opus-4-7", 100, 1_000_000))


def test_record_increments_spent():
    budget = ScopeBudget(limit_usd=10.00)
    budget.record({
        "model": "claude-sonnet-4-6",
        "input_tokens": 1_000_000,
        "output_tokens": 0,
    })
    assert budget.spent_usd == pytest.approx(3.00)


def test_record_accumulates_across_calls():
    budget = ScopeBudget(limit_usd=10.00)
    budget.record({"model": "claude-sonnet-4-6", "input_tokens": 1_000_000, "output_tokens": 0})
    budget.record({"model": "claude-sonnet-4-6", "input_tokens": 1_000_000, "output_tokens": 0})
    assert budget.spent_usd == pytest.approx(6.00)


def test_record_handles_cache_tokens():
    budget = ScopeBudget(limit_usd=10.00)
    budget.record({
        "model": "claude-sonnet-4-6",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
    })
    # 0.30 (read) + 3.75 (write) = 4.05
    assert budget.spent_usd == pytest.approx(4.05)


def test_budget_exceeded_is_exception():
    # Subclass check — callers may try/except on the bare class.
    assert issubclass(BudgetExceeded, Exception)
