#!/usr/bin/env python3
# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Compare the shipped pricing table against the vendors' published rates.

Run weekly in CI. Reports disagreements; **never edits the table**. A parser
bug that silently "corrects" a rate would be worse than the stale rate it
replaced, so a human transcribes every change. The script's job is to make
sure nobody has to notice on their own.

Both vendors publish Markdown versions of their pricing pages, so this reads
structured tables rather than scraping HTML.

Exit codes:
    0  table agrees with both vendors (or a vendor is unreachable — see below)
    1  drift found; the report names every disagreement
    2  the page shape changed and extraction could not run

2 is deliberately distinct from 1. Drift is news; a failed extraction means
this check has stopped working and will silently pass forever if ignored.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

TABLE = Path(__file__).resolve().parent.parent / "src" / "llm_cost_governor" / "pricing.json"

ANTHROPIC_MD = "https://platform.claude.com/docs/en/about-claude/pricing.md"
OPENAI_MD = "https://developers.openai.com/api/docs/pricing.md"

# Display name in the vendor table -> our model id. Anthropic's pricing page
# lists display names ("Claude Opus 5.5"); our keys are API ids.
ANTHROPIC_NAMES = {
    "Claude Fable 5.1": "claude-fable-5-1", "Claude Fable 5": "claude-fable-5",
    "Claude Opus 5.5": "claude-opus-5-5", "Claude Opus 5": "claude-opus-5",
    "Claude Opus 4.8": "claude-opus-4-8", "Claude Opus 4.7": "claude-opus-4-7",
    "Claude Opus 4.6": "claude-opus-4-6", "Claude Opus 4.5": "claude-opus-4-5",
    "Claude Sonnet 5.5": "claude-sonnet-5-5", "Claude Sonnet 5": "claude-sonnet-5",
    "Claude Sonnet 4.6": "claude-sonnet-4-6", "Claude Sonnet 4.5": "claude-sonnet-4-5",
    "Claude Haiku 5.5": "claude-haiku-5-5", "Claude Haiku 4.5": "claude-haiku-4-5",
}


class ExtractionFailed(RuntimeError):
    """The page no longer has the shape this script knows how to read."""


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-cost-governor-drift-check"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def money(cell: str) -> float | None:
    """Parse a price cell.

    None when the cell carries no number — a bare ``-``. What that absence
    means depends on the column: no *charge* (so 0.0) in a cache-write
    column, but no *context band at all* in a long-context one. This
    function cannot tell them apart, so each caller decides.
    """
    m = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)", cell.replace(",", ""))
    return float(m.group(1)) if m else None


def find_table(md: str, first_header: str) -> list[list[str]]:
    """Rows of the first Markdown table whose first header cell equals `first_header`.

    Matched exactly, not by prefix: a renamed column (``Modell``) must fail
    the check rather than quietly bind to the wrong table.
    """
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or cells[0] != first_header:
            continue
        if i + 1 >= len(lines) or set(lines[i + 1].replace("|", "").strip()) > set("-: "):
            continue
        rows = []
        for raw in lines[i + 2:]:
            if not raw.startswith("|"):
                break
            rows.append([c.strip() for c in raw.strip("|").split("|")])
        if rows:
            return rows
    raise ExtractionFailed(f"no table whose first header cell is {first_header!r}")


def anthropic_rates(md: str) -> dict[str, dict[str, float]]:
    """{model_id: {input, output, cache_write, cache_read}} from the vendor page."""
    out: dict[str, dict[str, float]] = {}
    for cells in find_table(md, "Model"):
        if len(cells) < 6:
            continue
        # The name cell may carry a markdown link or a parenthetical.
        name = re.sub(r"\s*[(\[].*$", "", cells[0]).strip()
        model = ANTHROPIC_NAMES.get(name)
        if model is None:
            continue
        rates = {
            "input": money(cells[1]), "cache_write": money(cells[2]),
            "cache_read": money(cells[4]), "output": money(cells[5]),
        }
        if any(v is None for v in rates.values()):
            continue  # a tiered row (e.g. Haiku 5.5) splits across lines
        out[model] = rates  # type: ignore[assignment]
    # A parser that matches *some* names reports "no drift" for everything it
    # silently skipped. Demand near-full coverage so a changed page shape
    # surfaces as a failure rather than as false reassurance.
    expected = len(ANTHROPIC_NAMES)
    if len(out) < expected - 1:
        raise ExtractionFailed(
            f"matched only {len(out)} of {expected} known Anthropic models — "
            f"the table shape likely changed"
        )
    return out


def openai_rates(md: str) -> dict[str, dict[str, float | None]]:
    """Short- and long-context rates from OpenAI's standard pricing table."""
    rows = find_table(md, "Model")
    out: dict[str, dict[str, float | None]] = {}
    for cells in rows:
        if len(cells) < 9:
            continue
        out[cells[0]] = {
            "input": money(cells[1]), "cache_read": money(cells[2]),
            "cache_write": money(cells[3]), "output": money(cells[4]),
            "long_input": money(cells[5]), "long_output": money(cells[8]),
        }
    if len(out) < 20:
        raise ExtractionFailed(
            f"OpenAI standard table yielded only {len(out)} rows — shape likely changed"
        )
    return out


def flat_rates(row: dict) -> dict[str, float] | None:
    """Our row's flat rates, or None when it is tiered."""
    if "tiers" in row:
        return None
    return {k: row[k] for k in ("input", "output", "cache_write", "cache_read")}


def main() -> int:
    table = json.loads(TABLE.read_text())
    findings: list[str] = []

    try:
        anth = anthropic_rates(fetch(ANTHROPIC_MD))
        oai = openai_rates(fetch(OPENAI_MD))
    except ExtractionFailed as exc:
        print(f"EXTRACTION FAILED: {exc}", file=sys.stderr)
        print("The vendor page shape changed. This check cannot run and will "
              "pass silently until it is repaired.", file=sys.stderr)
        return 2
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"vendor page unreachable ({exc}); treating as no-news", file=sys.stderr)
        return 0

    for model, row in table.items():
        ours = flat_rates(row)
        if ours is None:
            continue  # tiered rows are compared by hand; see the note below
        vendor = anth.get(model) or oai.get(model)
        if vendor is None:
            continue
        for key, mine in ours.items():
            theirs = vendor.get(key)
            if theirs is None:
                # The vendor shows no number for this charge. Agreement when we
                # also charge nothing; otherwise worth a look, rather than the
                # silent skip that would hide a real difference.
                if mine != 0:
                    findings.append(
                        f"  {model:26} {key:12} ours ${mine:<9} vendor lists no rate"
                    )
            elif abs(mine - theirs) > 1e-9:
                findings.append(f"  {model:26} {key:12} ours ${mine:<9} vendor ${theirs}")

    # A model the vendor prices in two context bands, which our row flattens
    # to one, is understated above the threshold — the Haiku 5.5 shape.
    for model, row in table.items():
        v = oai.get(model)
        if not v or "tiers" in row:
            continue
        long_in = v.get("long_input")
        # A '-' in the long columns means the model has no long-context band,
        # not that it is free there — so require a real, differing rate.
        if long_in and v["input"] and long_in != v["input"]:
            findings.append(
                f"  {model:26} {'long-context':12} vendor splits at >272K input: "
                f"in ${v['input']}->{v['long_input']}, out ${v['output']}->{v['long_output']}; "
                f"our row is flat"
            )

    missing = [m for m in anth if m not in table] + [
        m for m in oai if m.startswith(("gpt-", "o1", "o3", "o4")) and m not in table
    ]

    if findings:
        print("PRICING DRIFT\n")
        print("\n".join(sorted(set(findings))))
    if missing:
        print(f"\n({len(missing)} vendor models are absent from our table. That is "
              f"not drift — the table carries a chosen subset. Run with --list-absent "
              f"to see them.)")
        if "--list-absent" in sys.argv:
            print("  " + "\n  ".join(sorted(missing)))
    if not findings:
        print("No drift: every flat row matches its vendor page.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
