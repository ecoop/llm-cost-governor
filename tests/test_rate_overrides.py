# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Caller-supplied rate overrides (#18 item 4).

The gap these close: the shipped table carries a fixed set of models, so a
caller whose model is not among them is either refused by
`RequirePricedModelHook` or billed $0 — and can only fix it by getting a
release out of this repo. These tests pin the behaviour that makes an
override actually usable, especially that a registered model is admitted by
the pre-flight gate. A rate you can register but still cannot call would be
worse than the gap.
"""

import json

import pytest

from llm_cost_governor import pricing
from llm_cost_governor.budget import build_budget_chain
from llm_cost_governor.pricing import (
    CHAT,
    InvalidRateRow,
    catalog,
    is_priced,
    load_rates,
    overridden_models,
    register_rates,
    reset_overrides,
)
from llm_cost_governor.wrapper import CallContext, TokenEstimate

_NEW = {
    "label": "Acme 000", "capability": "chat", "provider": "acme",
    "input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01,
}


@pytest.fixture(autouse=True)
def _clean():
    reset_overrides()
    yield
    reset_overrides()


def _ctx(model):
    return CallContext(provider="anthropic", model=model, kwargs={}, tags={},
                       estimate=TokenEstimate(input_tokens=1_000, output_tokens=1_000))


# ── a model the library doesn't carry ──────────────────────────────────────────

def test_registering_a_new_model_prices_it():
    assert not is_priced("acme-does-not-exist-000")
    register_rates({"acme-does-not-exist-000": _NEW})
    assert is_priced("acme-does-not-exist-000")
    assert pricing._cost("acme-does-not-exist-000", 1_000_000, 1_000_000) == pytest.approx(0.60)


def test_registered_model_is_admitted_by_the_preflight_gate():
    # The failure mode that would make this feature worse than useless:
    # priced, but still refused. RATES and MODEL_PRICING must move together.
    register_rates({"acme-does-not-exist-000": _NEW})
    for hook in build_budget_chain(limit_usd=100.00):
        hook.pre(_ctx("acme-does-not-exist-000"))      # must not raise


def test_registered_model_appears_in_catalog():
    # Consumers filter the catalog to build model pickers; a registered model
    # that prices but doesn't list is invisible where it matters.
    register_rates({"acme-does-not-exist-000": _NEW})
    assert "acme-does-not-exist-000" in {m.id for m in catalog(CHAT)}


# ── correcting a model the library carries wrongly ─────────────────────────────

def test_partial_override_keeps_the_rest_of_the_row():
    before = dict(pricing.MODEL_PRICING["claude-opus-5"])
    register_rates({"claude-opus-5": {"input": 1.23}})
    assert pricing.RATES["claude-opus-5"]["input"] == 1.23
    assert pricing.MODEL_PRICING["claude-opus-5"]["label"] == before["label"]
    assert pricing.MODEL_PRICING["claude-opus-5"]["output"] == before["output"]


def test_reset_restores_the_shipped_row_exactly():
    shipped = dict(pricing.MODEL_PRICING["claude-opus-5"])
    register_rates({"claude-opus-5": {"input": 999.0}})
    reset_overrides()
    assert pricing.MODEL_PRICING["claude-opus-5"] == shipped
    assert pricing.RATES["claude-opus-5"]["input"] == shipped["input"]


# ── refusing to half-apply ─────────────────────────────────────────────────────

def test_new_model_needs_a_complete_record():
    with pytest.raises(InvalidRateRow, match="complete record"):
        register_rates({"acme-1": {"input": 1.0, "output": 2.0}})


@pytest.mark.parametrize("bad", [
    {"input": -1.0}, {"input": "cheap"}, {"output": True},
])
def test_malformed_rates_are_refused(bad):
    with pytest.raises(InvalidRateRow):
        register_rates({"claude-opus-5": bad})


def test_a_bad_row_applies_none_of_the_batch():
    # A half-applied override is a silently wrong price, which is the whole
    # failure class this module exists to prevent.
    before = pricing.RATES["claude-opus-5"]["input"]
    with pytest.raises(InvalidRateRow):
        register_rates({
            "claude-opus-5": {"input": 1.11},     # valid
            "acme-1": {"input": 1.0},             # invalid: new, incomplete
        })
    assert pricing.RATES["claude-opus-5"]["input"] == before
    assert not overridden_models()


# ── visibility ─────────────────────────────────────────────────────────────────

def test_overrides_are_reported():
    register_rates({"acme-does-not-exist-000": _NEW, "claude-opus-5": {"input": 1.0}})
    assert overridden_models() == frozenset({"acme-does-not-exist-000", "claude-opus-5"})


def test_overrides_are_excluded_from_freshness():
    # RATES_AS_OF asserts "a human checked this vendor's page". The library has
    # no basis to call a caller-supplied rate fresh or stale, so registering
    # one must not move the provenance clock in either direction.
    before = dict(pricing.RATES_AS_OF)
    register_rates({"acme-does-not-exist-000": _NEW})
    assert dict(pricing.RATES_AS_OF) == before


# ── the file form ──────────────────────────────────────────────────────────────

def test_load_rates_from_a_json_path(tmp_path):
    f = tmp_path / "rates.json"
    f.write_text(json.dumps({"acme-does-not-exist-000": _NEW}))
    load_rates(str(f))
    assert is_priced("acme-does-not-exist-000")


def test_load_rates_from_a_json_string():
    load_rates(json.dumps({"acme-does-not-exist-000": _NEW}))
    assert is_priced("acme-does-not-exist-000")


def test_load_rates_through_a_state_backend():
    # Reuses the abstraction already used for counter state, so a rates file
    # can live in the same GCS bucket and change without rebuilding an image.
    class _Backend:
        def read(self, object_name):
            return json.dumps({"acme-does-not-exist-000": _NEW}) if object_name == "rates.json" else None
        def write(self, object_name, text): ...

    load_rates(_Backend())
    assert is_priced("acme-does-not-exist-000")


def test_load_rates_missing_backend_object_is_an_error():
    class _Empty:
        def read(self, object_name): return None
        def write(self, object_name, text): ...

    with pytest.raises(InvalidRateRow, match="no rates object"):
        load_rates(_Empty())


def test_load_rates_rejects_a_non_object_document():
    with pytest.raises(InvalidRateRow, match="expected an object"):
        load_rates("[1, 2, 3]")
