# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Provider adapters — one module per LLM provider.

Each adapter module implements a small module-level API:

    call(client, kwargs)          -> Any    # invoke the provider SDK
    extract_usage(response)       -> dict   # normalized token dict
    truncated(response)           -> bool   # was max_tokens hit?
    default_estimate(kwargs)      -> TokenEstimate

`get_provider(name)` returns the adapter module by name; unknown names
raise ValueError so a typo surfaces at first call instead of silently
degrading. The registry starts with Anthropic only — additional
providers (OpenAI, Voyage) are added as new modules with a new registry
entry, no changes needed anywhere in the wrapper.
"""

from __future__ import annotations

from types import ModuleType


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
    raise ValueError(
        f"Unknown provider {name!r}; expected one of: 'anthropic'."
    )
