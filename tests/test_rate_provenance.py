# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Rate-provenance freshness (#18).

A stale rate is the one pricing failure nothing else can catch. An *absent*
row is loud — `_warn_unpriced` fires and `RequirePricedModelHook` refuses the
call. A *wrong* row is silent: `is_priced()` returns True, the gate admits the
call, and every budget and cap downstream is computed from a bad figure with
no signal at all.

The library cannot tell that a number is wrong — no vendor publishes a
machine-readable price feed. What it can do is track how long it has been
since a human checked, and make that a build failure on a deadline.
"""

import warnings
from datetime import date, timedelta

import pytest

from llm_cost_governor.pricing import (
    PROVIDERS,
    RATE_SOURCES,
    RATES_AS_OF,
    RATES_STALE_FAIL_DAYS,
    RATES_STALE_WARN_DAYS,
    rates_age_days,
    stalest_rates,
)

# ── the provenance table itself ────────────────────────────────────────────────

def test_every_provider_has_recorded_provenance():
    # A provider whose rates have no date is one nothing will ever chase.
    missing = set(PROVIDERS) - set(RATES_AS_OF)
    assert not missing, f"providers with no rates-as-of date: {sorted(missing)}"


def test_every_provenance_entry_has_a_source_url():
    # The date is only actionable next to the page you re-check against.
    assert set(RATES_AS_OF) == set(RATE_SOURCES), (
        f"RATES_AS_OF and RATE_SOURCES disagree: "
        f"{set(RATES_AS_OF) ^ set(RATE_SOURCES)}"
    )
    for source, url in RATE_SOURCES.items():
        assert url.startswith("https://"), f"{source} has no usable source URL"


def test_no_provenance_date_is_in_the_future():
    today = date.today()  # noqa: DTZ011 — a day's drift cannot matter here
    ahead = {s: d for s, d in RATES_AS_OF.items() if d > today + timedelta(days=1)}
    assert not ahead, f"rates dated in the future (typo?): {ahead}"


# ── the deadline ───────────────────────────────────────────────────────────────

def test_rate_provenance_is_fresh():
    """Fails once any vendor's rates have gone unchecked past the limit.

    This test is *supposed* to fail eventually — that is the mechanism, not a
    defect. When it does: open each source URL, confirm or correct the rates,
    and bump that entry's date in `RATES_AS_OF` in the same commit.

    `RATES_STALE_WARN_DAYS` fires a warning first, so the deadline is visible
    for ~3 months before it can redden an unrelated PR.
    """
    stale = {
        s: rates_age_days(s)
        for s in RATES_AS_OF
        if rates_age_days(s) > RATES_STALE_FAIL_DAYS
    }
    if stale:
        lines = "\n".join(
            f"    {s}: {age} days old — re-check {RATE_SOURCES[s]}"
            for s, age in sorted(stale.items(), key=lambda kv: -kv[1])
        )
        pytest.fail(
            f"Vendor rates unchecked for more than {RATES_STALE_FAIL_DAYS} days:\n"
            f"{lines}\n"
            f"  A wrong rate is silent — is_priced() still returns True, so the "
            f"pre-flight gate admits the call and budgets use the bad figure.\n"
            f"  Fix: verify against the URL above, then bump RATES_AS_OF in the "
            f"same commit as any rate change."
        )

    warn = {
        s: rates_age_days(s)
        for s in RATES_AS_OF
        if rates_age_days(s) > RATES_STALE_WARN_DAYS
    }
    if warn:
        warnings.warn(
            f"Vendor rates due a re-check (>{RATES_STALE_WARN_DAYS} days): {warn}",
            UserWarning,
            stacklevel=2,
        )


def test_the_deadline_actually_trips():
    # Guards the guard: a freshness test that can never fail is decoration.
    # Age every entry past the fail threshold and confirm it would fire.
    future = date.today() + timedelta(days=RATES_STALE_FAIL_DAYS + 1)  # noqa: DTZ011
    aged = {s: rates_age_days(s, today=future) for s in RATES_AS_OF}
    assert all(age > RATES_STALE_FAIL_DAYS for age in aged.values()), aged


# ── the public accessors ───────────────────────────────────────────────────────

def test_rates_age_days_accepts_an_injected_today():
    src = next(iter(RATES_AS_OF))
    on_the_day = rates_age_days(src, today=RATES_AS_OF[src])
    assert on_the_day == 0


def test_rates_age_days_raises_for_an_unknown_source():
    with pytest.raises(KeyError):
        rates_age_days("not-a-vendor")


def test_stalest_rates_reports_the_oldest():
    src, age = stalest_rates()
    assert src in RATES_AS_OF
    assert age == max(rates_age_days(s) for s in RATES_AS_OF)
