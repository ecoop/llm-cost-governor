# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Provider adapters — one module per LLM provider.

Each adapter module implements a small module-level API:

    call(client, kwargs)          -> Any    # invoke the provider SDK
    extract_usage(response)       -> dict   # normalized token dict
    truncated(response)           -> bool   # was max_tokens hit?
    default_estimate(kwargs)      -> TokenEstimate

`get_provider(name)` returns the adapter module by name; unknown names
raise ValueError so a typo surfaces at first call instead of silently
degrading.

`ADAPTERS` names every provider with an adapter, and is the set
`get_provider` accepts. **It is deliberately not the same as the set of
providers in the pricing table.** Pricing data and `guarded_call`
support are separable: `record_usage` prices any model in
`MODEL_PRICING` without an adapter, which is how Voyage embeddings and
reranks have always been metered. A rate row means "we can cost this";
an adapter means "we can also wrap the call and run pre-flight hooks".
Requiring parity between them would break the `record_usage`-only
providers that work by design today.
"""

from __future__ import annotations

from types import ModuleType

# Providers with an adapter module — exactly what `get_provider` accepts.
# A new adapter must be added here and to `get_provider` together;
# `test_adapters_declaration_matches_get_provider` fails if they drift.
ADAPTERS: frozenset[str] = frozenset({"anthropic", "openai"})


def get_provider(name: str) -> ModuleType:
    """Return the provider adapter module by name.

    Raises:
        ValueError: When ``name`` doesn't match a known provider.
            Deferred to first call site so a typo in a caller surfaces
            immediately rather than dropping to a silent default.
    """
    if name == "anthropic":
        from llm_cost_governor.providers import anthropic
        return anthropic
    if name == "openai":
        from llm_cost_governor.providers import openai
        return openai
    raise ValueError(
        f"Unknown provider {name!r}; expected one of: "
        f"{', '.join(repr(p) for p in sorted(ADAPTERS))}."
    )
