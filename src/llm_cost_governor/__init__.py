# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""llm-cost-governor — pricing, budgets, cost caps, rate limits, observability.

A standalone, framework-agnostic library of composable pre-call and
post-call hooks for LLM API calls. It carries no coupling to any host
application: app-specific concerns (alerts, invite identity, LangSmith
metadata) enter via caller-registered callbacks and hooks, and every
integration (FastAPI, GCS, OTel) lives behind an optional extra.
"""
