# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""LLM guardrails — pricing, budgets, cost caps, rate limits, observability.

Working-title internal package that will eventually extract into its own
repo (see the extraction discussion). Currently vendored inside pitchcraft.
Every module here must remain free of imports from the rest of pitchcraft;
app-specific concerns (alerts, invite identity, LangSmith metadata) enter
via caller-registered callbacks and hooks.
"""
