# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Unit tests for scripts/check_pricing_drift.py.

The drift check is only useful if it fails loudly when a vendor page changes
shape. A parser that quietly matches nothing reports "no drift" for the entire
table — false reassurance, which is worse than no check. These tests run
against fixture Markdown; nothing here touches the network.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_pricing_drift import (
    ANTHROPIC_NAMES,
    ExtractionFailed,
    anthropic_rates,
    find_table,
    money,
    openai_rates,
)

_ANTHROPIC_MD = """# Pricing

## Model pricing

| Model | Base input tokens | 5m cache writes | 1h cache writes | Cache hits and refreshes | Output tokens |
| :--- | :--- | :--- | :--- | :--- | :--- |
""" + "\n".join(
    f"| {name} | $5 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok | $25 / MTok |"
    for name in ANTHROPIC_NAMES
) + "\n"

_OPENAI_MD = """# Pricing

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(
    f"| model-{i} | $2.00 | $0.20 | $2.50 | $10.00 | $4.00 | $0.40 | $5.00 | $15.00 |"
    for i in range(25)
) + "\n"


# ── cell parsing ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("cell", "want"), [
    ("$5 / MTok", 5.0), ("$0.125 / MTok", 0.125), ("$1,234 / MTok", 1234.0),
    ("  $10.00  ", 10.0), ("n/a", None), ("", None), ("Not listed", None),
])
def test_money_parses_price_cells(cell, want):
    assert money(cell) == want


# ── the loud-failure contract ──────────────────────────────────────────────────

def test_a_renamed_header_fails_rather_than_matching_nothing():
    with pytest.raises(ExtractionFailed, match="no table whose first header"):
        find_table("| Modell | Price |\n| --- | --- |\n| a | $1 |\n", "Model")


def test_partial_anthropic_coverage_is_an_extraction_failure():
    # The dangerous case: the page still parses, but most rows no longer match,
    # so every unmatched model is silently reported as "no drift".
    lines = _ANTHROPIC_MD.splitlines()
    trimmed = "\n".join(lines[:6] + lines[6:8]) + "\n"   # header, rule, 2 rows
    with pytest.raises(ExtractionFailed, match="matched only"):
        anthropic_rates(trimmed)


def test_too_few_openai_rows_is_an_extraction_failure():
    short = "\n".join(_OPENAI_MD.splitlines()[:8]) + "\n"
    with pytest.raises(ExtractionFailed, match="only"):
        openai_rates(short)


# ── happy paths ────────────────────────────────────────────────────────────────

def test_anthropic_rows_map_display_names_to_model_ids():
    rates = anthropic_rates(_ANTHROPIC_MD)
    assert set(rates) == set(ANTHROPIC_NAMES.values())
    assert rates["claude-opus-5"] == {
        "input": 5.0, "cache_write": 6.25, "cache_read": 0.50, "output": 25.0,
    }


def test_anthropic_name_cell_tolerates_links_and_parentheticals():
    md = _ANTHROPIC_MD.replace(
        "| Claude Opus 5 |",
        "| Claude Opus 5 ([retired](https://example.com/x)) |",
    )
    assert "claude-opus-5" in anthropic_rates(md)


def test_openai_parses_both_context_bands():
    rates = openai_rates(_OPENAI_MD)
    row = rates["model-0"]
    assert row["input"] == 2.00
    assert row["long_input"] == 4.00      # the band our flat rows don't model
    assert row["output"] == 10.00
    assert row["long_output"] == 15.00


def test_every_mapped_anthropic_name_is_a_real_model_id():
    # A typo in the name map would silently exclude that model from the check.
    from llm_cost_governor.pricing import MODEL_PRICING

    unknown = [m for m in ANTHROPIC_NAMES.values() if m not in MODEL_PRICING]
    assert not unknown, f"drift check maps to ids absent from the table: {unknown}"
