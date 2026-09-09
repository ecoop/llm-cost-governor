# llm-cost-governor

[![PyPI](https://img.shields.io/pypi/v/llm-cost-governor)](https://pypi.org/project/llm-cost-governor/)
[![CI](https://github.com/ecoop/llm-cost-governor/actions/workflows/ci.yml/badge.svg)](https://github.com/ecoop/llm-cost-governor/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/llm-cost-governor)](https://pypi.org/project/llm-cost-governor/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Composable pre-call and post-call hooks for LLM API calls: **pricing, budgets, cost caps, rate limits, event log, observability**.

Wrap your existing Anthropic or OpenAI SDK calls with a single `guarded_call(client, ...)`, register the hooks you need, and get:

- **Priced cost** per call from a shared pricing table — 38 models across Anthropic, OpenAI and Voyage, each tagged with its provider and capability.
- **Session or scope budgets** with pre-flight enforcement.
- **Rolling-window cost caps** (hourly / daily / weekly, optionally per-identity) with durable state (local disk or GCS).
- **Per-IP request rate limiting** as a FastAPI dependency factory.
- **Structured event log** (one JSON line per call) for offline analysis.
- **OpenTelemetry span** per call, with LangSmith metadata support and per-request content scrubbing.
- **Framework-agnostic core** — the FastAPI, GCS, and OTel bits are optional extras. Zero coupling to any host application.

The library was extracted from Pitchcraft and is used in production by three apps; one of them, [Rulebook](https://github.com/ecoop/rulebook), is public.

**Adopting this in a new app?** See [`docs/integration.md`](docs/integration.md) for the DI pattern, FastAPI init-order gotcha, constructor signatures, and reference implementation.

---

## Install

```bash
# Core install
pip install llm-cost-governor

# With optional integrations
pip install "llm-cost-governor[fastapi,gcs,otel]"
```

Requires Python 3.11+. The core has just one dependency (pydantic v2); every integration is behind an optional extra so the install stays lean.

---

## Quick example

```python
from pathlib import Path

from anthropic import Anthropic
from llm_cost_governor.budget import (
    RequirePricedModelHook, ScopeBudget, ScopeBudgetHook,
)
from llm_cost_governor.counters import CostCounter, WindowedCapHook
from llm_cost_governor.events import EventLogHook
from llm_cost_governor.state import LocalFileBackend
from llm_cost_governor.wrapper import guarded_call

client = Anthropic()

# Wire up the counter at startup — one instance, shared across requests.
counter = CostCounter(
    object_name="cost_counter.json",
    backend=LocalFileBackend(root=Path("./state")),
    enabled=True,
    hourly_cap_usd=0.50, daily_cap_usd=2.00,
    weekly_cap_usd=10.00, per_token_cap_usd=1.00,
)
counter.load()

# Per-request: build a fresh scope budget, compose the hook chain.
budget = ScopeBudget(limit_usd=0.25)
hooks = [
    RequirePricedModelHook(),      # refuse any model the pricing table doesn't know
    ScopeBudgetHook(budget),
    WindowedCapHook(counter),
    EventLogHook(enabled=True),
]

# The one line that replaces `client.messages.create(...)`.
response, usage = guarded_call(
    client,
    provider="anthropic",
    hooks=hooks,
    tags={"stage": "drafter"},
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=512,
)

print(f"Cost: ${usage.cost_usd:.4f}   Tokens: {usage.input_tokens} in / {usage.output_tokens} out")
```

That's it. Every hook's `pre` runs before the SDK call (aborts on `UnpricedModel` / `BudgetExceeded` / `CostCapExceeded` / `RateLimitExceeded`); every `post` runs after with the priced `UsageRecord` and updates the shared state.

---

## Core concepts

### The Hook chain

`guarded_call(client, ..., hooks=[...])` runs each hook's `pre(ctx)` method before the SDK call and each `post(ctx, usage)` after. A hook is any object with those two methods and a `name` attribute — implement your own by satisfying the [`Hook`](src/llm_cost_governor/wrapper.py) Protocol. The shipped hooks:

| Hook | pre | post |
|---|---|---|
| `RequirePricedModelHook` | raise `UnpricedModel` if the model has no rate row | no-op |
| `ScopeBudgetHook` | raise `BudgetExceeded` if the pre-flight estimate would push over | record the actual cost against the budget |
| `WindowedCapHook` | raise `CostCapExceeded` if a rolling window is already at cap | record cost + trigger alerts on cap crossings |
| `ProviderTotalsHook` | no-op | add the call's cost to its provider's running total (the per-provider breakdown `CostCounter` aggregates away) |
| `EventLogHook` | no-op | emit one structured JSON line to stdout |
| `OTelSpanHook` (optional) | open a span with `gen_ai.request.*` attrs | close it with `gen_ai.usage.*` + cost attrs |
| `LangSmithMetadataHook` (optional) | stamp `langsmith.metadata.*` from a caller-supplied identity dict | no-op |

### Providers

`guarded_call(provider="anthropic" | "openai", ...)` selects the adapter that knows how to invoke the SDK and normalize its response. Both ship in-box; further adapters slot in as new modules with a couple lines each — see [`providers/anthropic.py`](src/llm_cost_governor/providers/anthropic.py) for the shape.

Adapters are passthroughs, not translation layers: each forwards your kwargs to that vendor's SDK unchanged. Normalizing one prompt shape across providers is an application concern, and pushing it in here would make a cost library into an API shim.

Not every priced model needs an adapter. `providers.ADAPTERS` names those that have one; Voyage models are priced and metered through `record_usage` below, which needs none.

### A gated ceiling in one call

The common case — a hard USD ceiling that can't be silently escaped — is one import:

```python
from llm_cost_governor.budget import build_budget_chain

hooks = build_budget_chain(limit_usd=3.00)   # [RequirePricedModelHook, ScopeBudgetHook]
```

Prefer it over assembling the pair by hand. An unpriced model costs `$0`, so it moves no
ledger and no ceiling can trip for it — the gate is what closes that, and a single import
that either resolves or raises is what stops an older install from silently degrading to
no enforcement at all.

**A budget's scope is its object's lifetime.** There's no scope argument and no per-scope
class: a `ScopeBudget` covers exactly the calls that share the instance. Widening the
ceiling means holding it longer and passing it further:

```python
sweep = ScopeBudget(limit_usd=50.00)          # one per sweep, not per call

for case in cases:
    hooks = build_budget_chain(budget=sweep)   # same instance every iteration
    guarded_call(client, hooks=hooks, **kwargs)

sweep.spent_usd                                # running sweep total
```

Budgets nest — a per-run and a per-sweep instance can sit in the same chain and enforce
independently. One caveat: a shared budget is **in-memory and process-local**, so it does
not span subprocesses, distributed workers, or a horizontally scaled service — each
process gets its own object, and the effective ceiling becomes N × processes. Bounding
spend across processes is what `CostCounter` on a shared `StateBackend` is for.

### State backends

Counters can persist their rolling-window state through the `StateBackend` Protocol. Two implementations ship:

- `LocalFileBackend(root)` — one JSON file per state object under a local directory, created on first write. Default for local dev / CI.
- `GcsBackend(bucket)` — Google Cloud Storage blob. Lazily imports `google-cloud-storage` on first use, so the core install stays dep-free.

Add your own by implementing `read(name) -> str | None` and `write(name, text) -> None`.

### `record_usage` — for calls you made yourself

Voyage embeddings, batch APIs, vision — anything that doesn't fit the `guarded_call` shape. `record_usage()` runs only the **post** hooks, still gives you priced cost and event log, without wrapping the call:

```python
from llm_cost_governor.wrapper import record_usage

response = voyage_client.embed(texts=[...], model="voyage-4")
record_usage(
    provider="voyage", model="voyage-4",
    input_tokens=response.total_tokens, output_tokens=0,
    hooks=hooks, tags={"call_type": "embedding"},
)
```

---

## What's in / what's out

**Included:**
- Pricing for 38 currently-shipped models — Claude (Fable 5, Opus 5, Sonnet 5, Opus 4.6/4.7/4.8, Sonnet 4.6, Haiku 4.5), OpenAI (GPT-5 family + embeddings), and Voyage (embeddings + rerank). Each row carries its provider and capability, so consumers can filter the catalog instead of pattern-matching model ids. Easy to extend as new models ship.
- Rolling-window counter with configurable caps + durable persistence.
- `RollingWeekCounter` — a generic per-key, rolling-week cumulative counter with cap enforcement (strict or lenient) across one or more named dimensions; the reusable core behind app-specific caps like per-token/per-IP upload limits. Import from `llm_cost_governor.counters`.
- `ProviderTotals` + `ProviderTotalsHook` — a per-provider cumulative-USD read-model (anthropic vs voyage vs …) for usage widgets and cost dashboards; the breakdown the windowed `CostCounter` aggregates away. In-memory by default, optionally persisted through a `StateBackend`. Import from `llm_cost_governor.provider_totals`.
- `RequirePricedModelHook` + `build_budget_chain()` — the pre-flight gate that refuses a model the pricing table can't cost, and the one-call helper that wires it in front of a budget. Without it an unpriced model is not billed loosely, it is exempt from every budget and cap.
- Rate provenance — `RATES_AS_OF` records when each vendor's prices were last checked, and a test fails the build once any goes unchecked too long. A *missing* rate is loud (the gate refuses the call); a *wrong* one is silent, and this is the only thing that catches it.
- Per-IP rate limiter (framework-neutral core + FastAPI dependency factory).
- Structured event log (stdout → any log aggregator).
- Provider adapters for Anthropic and OpenAI.
- OTel span hooks + a request-span context manager + LangSmith metadata.
- Content-scrubbing OTel exporter for per-request telemetry control.
- Discord-webhook alert sink (implements the `AlertSink` Protocol).

**Not (yet) included:**
- Provider adapters for Voyage and Gemini — the shape is fixed and each is ~30 lines, but they're not in the box until someone needs them. (Voyage models are priced; they're metered via `record_usage` rather than `guarded_call`.)
- Streaming responses. The wrapper is synchronous today; adding async is straightforward but not implemented yet.
- Multi-instance atomic counters — the rolling-window counter is correct at `max-instances=1`. Distributed correctness (e.g., Redis-backed) is a future extension.
- Retry / circuit-breaker logic. The library never retries — that's the caller's responsibility.

---

## Development

```bash
git clone https://github.com/ecoop/llm-cost-governor
cd llm-cost-governor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check src tests
```

CI runs on Python 3.11, 3.12, 3.13 via [GitHub Actions](.github/workflows/ci.yml).

## Versioning

Currently `v0.4.6`. Published to PyPI since `0.3.0`.

| | |
|---|---|
| `0.3.x` | renamed the distribution from `llm-guardrails` to `llm-cost-governor` |
| `0.4.0` | `capability` on every model record, plus `catalog()` |
| `0.4.1` | `RequirePricedModelHook` — the pre-flight gate; OpenAI GPT-5 and embedding rates |
| `0.4.2` | meters Anthropic server-side tool use (web search) instead of dropping it |
| `0.4.3` | OpenAI provider adapter, and real OpenAI cache rates |
| `0.4.4` | `provider` on every model record |
| `0.4.5` | `build_budget_chain()` — a gated ceiling in one import, or a loud `ImportError` |
| `0.4.6` | rate provenance — `RATES_AS_OF`, plus a build failure when it goes stale |

Semver from `v1.0.0` onward; anything before is "shipped but pre-stable API — expect breaking changes."

## Contributing

Issues and pull requests welcome. For substantive changes, open an issue first to discuss the shape before writing code. The Hook Protocol and provider-adapter surface are the two most important extension points — happy to talk through how to add a new provider or hook.

## License

MIT. See [LICENSE](LICENSE).

---

_Last updated:_ 2026-09-09
