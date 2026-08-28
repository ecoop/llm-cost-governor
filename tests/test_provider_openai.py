# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Unit tests for the OpenAI provider adapter and the ADAPTERS declaration."""

import pytest

from llm_cost_governor.pricing import _cost
from llm_cost_governor.providers import ADAPTERS, get_provider
from llm_cost_governor.providers import openai as adapter


class _Details:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, details=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if details is not None:
            self.prompt_tokens_details = details


class _Choice:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, usage=None, finish_reason="stop"):
        self.usage = usage or _Usage()
        self.choices = [_Choice(finish_reason)]


# ── registration ───────────────────────────────────────────────────────────────

def test_get_provider_returns_the_openai_adapter():
    assert get_provider("openai") is adapter


def test_adapters_declaration_matches_get_provider():
    # The declaration and the dispatch must not drift: every name in ADAPTERS
    # must resolve, and anything outside it must raise.
    for name in ADAPTERS:
        assert get_provider(name) is not None
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("not-a-provider")


def test_adapters_is_not_required_to_match_the_pricing_table():
    # Documents the deliberate asymmetry. Voyage/rerank rows are metered via
    # record_usage and have no adapter by design; a parity check between
    # MODEL_PRICING and ADAPTERS would fail on working, intended state.
    from llm_cost_governor.pricing import MODEL_PRICING

    voyage = [m for m in MODEL_PRICING if m.startswith(("voyage-", "rerank-"))]
    assert voyage, "expected Voyage/rerank rows"
    assert "voyage" not in ADAPTERS


# ── the four adapter functions ─────────────────────────────────────────────────

def test_call_forwards_to_chat_completions(monkeypatch):
    seen = {}

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    seen.update(kwargs)
                    return "resp"

    assert adapter.call(_Client(), {"model": "gpt-5", "messages": []}) == "resp"
    assert seen == {"model": "gpt-5", "messages": []}


def test_extract_usage_maps_openai_field_names():
    r = _Response(_Usage(prompt_tokens=100, completion_tokens=40))
    u = adapter.extract_usage(r)
    assert u["input_tokens"] == 100
    assert u["output_tokens"] == 40


def test_cached_tokens_are_subtracted_from_input():
    # OpenAI reports cached_tokens *inside* prompt_tokens. Leaving them in
    # would bill the same tokens twice — once at full input, again at the
    # cached rate.
    r = _Response(_Usage(prompt_tokens=1000, completion_tokens=0,
                         details=_Details(cached_tokens=800)))
    u = adapter.extract_usage(r)
    assert u["input_tokens"] == 200
    assert u["cache_read_input_tokens"] == 800


def test_cache_creation_is_none_not_zero():
    # OpenAI caching is automatic — there is no creation step to account for.
    # None means "not exposed"; 0 would claim the provider reported zero.
    r = _Response(_Usage(prompt_tokens=10, completion_tokens=1))
    assert adapter.extract_usage(r)["cache_creation_input_tokens"] is None


def test_cached_tokens_none_when_details_absent():
    r = _Response(_Usage(prompt_tokens=10, completion_tokens=1))
    assert adapter.extract_usage(r)["cache_read_input_tokens"] is None


def test_details_as_a_dict_also_works():
    r = _Response(_Usage(prompt_tokens=100, completion_tokens=0,
                         details={"cached_tokens": 25}))
    assert adapter.extract_usage(r)["cache_read_input_tokens"] == 25


def test_truncated_on_length_finish_reason():
    assert adapter.truncated(_Response(finish_reason="length")) is True
    assert adapter.truncated(_Response(finish_reason="stop")) is False


def test_truncated_handles_empty_choices():
    r = _Response()
    r.choices = []
    assert adapter.truncated(r) is False


def test_default_estimate_accepts_both_ceiling_spellings():
    assert adapter.default_estimate({"max_completion_tokens": 500}).output_tokens == 500
    assert adapter.default_estimate({"max_tokens": 300}).output_tokens == 300
    assert adapter.default_estimate({}).output_tokens == 1024


# ── the cache rates the adapter made load-bearing ──────────────────────────────

def test_cached_openai_tokens_are_priced_not_free():
    # Before 0.4.3 the OpenAI rows carried cache_read = 0.00, which was safe
    # only while nothing extracted cached tokens. The adapter changed that:
    # a cached call must cost the discounted rate, not $0.
    cost = _cost("gpt-5", input_tok=0, output_tok=0, cache_read_tok=1_000_000)
    assert cost == pytest.approx(0.125), "cached OpenAI input must not be free"


def test_pro_models_have_no_cache_rates():
    # The -pro models list no cached-input rate: they don't support caching.
    from llm_cost_governor.pricing import MODEL_PRICING

    for model in [m for m in MODEL_PRICING if m.startswith("gpt-") and m.endswith("-pro")]:
        row = MODEL_PRICING[model]
        assert row["cache_read"] == 0.0 and row["cache_write"] == 0.0
