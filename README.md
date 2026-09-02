# llm-cost-governor

Composable pre-call and post-call hooks for LLM API calls: **pricing, budgets, cost caps, rate limits, event log, observability**.

Wrap your existing Anthropic / OpenAI / Voyage SDK calls with a single `guarded_call(client, ...)`, register the hooks you need, and get:

- **Priced cost** per call from a shared pricing table (Sonnet, Opus, Haiku, GPT-4/5, Voyage embeddings, easy to extend).
- **Session or scope budgets** with pre-flight enforcement.
- **Rolling-window cost caps** (hourly / daily / weekly, optionally per-identity) with durable state (local disk or GCS).
- **Per-IP request rate limiting** as a FastAPI dependency factory.
- **Structured event log** (one JSON line per call) for offline analysis.
- **OpenTelemetry span** per call, with LangSmith metadata support and per-request content scrubbing.
- **Framework-agnostic core** — the FastAPI, GCS, and OTel bits are optional extras. Zero coupling to any host application.

The library was extracted from [Pitchcraft](https://github.com/ecoop/pitchcraft) and is currently consumed there in production; a second consumer ([Rulebook](https://github.com/ecoop/rulebook)) is scheduled to adopt it.

**Adopting this in a new app?** See [`docs/integration.md`](docs/integration.md) for the DI pattern, FastAPI init-order gotcha, constructor signatures, and reference implementation.

---

## Install

```bash
# Core install
pip install "llm-cost-governor @ git+https://github.com/ecoop/llm-cost-governor@v0.3.0"

# With optional integrations
pip install "llm-cost-governor[fastapi,gcs,otel] @ git+https://github.com/ecoop/llm-cost-governor@v0.3.0"
```

Requires Python 3.11+. The core has just one dependency (pydantic v2); every integration is behind an optional extra so the install stays lean.

---

## Quick example

```python
from anthropic import Anthropic
from llm_cost_governor.wrapper import guarded_call
from llm_cost_governor.budget import ScopeBudget, ScopeBudgetHook
from llm_cost_governor.counters import CostCounter, WindowedCapHook
from llm_cost_governor.events import EventLogHook
from llm_cost_governor.state import LocalFileBackend

client = Anthropic()

# Wire up the counter at startup — one instance, shared across requests.
counter = CostCounter(
    object_name="cost_counter.json",
    backend=LocalFileBackend(path="./state"),
    enabled=True,
    hourly_cap_usd=0.50, daily_cap_usd=2.00,
    weekly_cap_usd=10.00, per_token_cap_usd=1.00,
)
counter.load()

# Per-request: build a fresh scope budget, compose the hook chain.
budget = ScopeBudget(limit_usd=0.25)
hooks = [
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

That's it. Every hook's `pre` runs before the SDK call (aborts on `BudgetExceeded` / `CostCapExceeded` / `RateLimitExceeded`); every `post` runs after with the priced `UsageRecord` and updates the shared state.

---

## Core concepts

### The Hook chain

`guarded_call(client, ..., hooks=[...])` runs each hook's `pre(ctx)` method before the SDK call and each `post(ctx, usage)` after. A hook is any object with those two methods and a `name` attribute — implement your own by satisfying the [`Hook`](src/llm_cost_governor/wrapper.py) Protocol. The shipped hooks:

| Hook | pre | post |
|---|---|---|
| `ScopeBudgetHook` | raise `BudgetExceeded` if the pre-flight estimate would push over | record the actual cost against the budget |
| `WindowedCapHook` | raise `CostCapExceeded` if a rolling window is already at cap | record cost + trigger alerts on cap crossings |
| `ProviderTotalsHook` | no-op | add the call's cost to its provider's running total (the per-provider breakdown `CostCounter` aggregates away) |
| `EventLogHook` | no-op | emit one structured JSON line to stdout |
| `OTelSpanHook` (optional) | open a span with `gen_ai.request.*` attrs | close it with `gen_ai.usage.*` + cost attrs |
| `LangSmithMetadataHook` (optional) | stamp `langsmith.metadata.*` from a caller-supplied identity dict | no-op |

### Providers

`guarded_call(provider="anthropic", ...)` selects the adapter that knows how to invoke the SDK and normalize the response. The Anthropic adapter ships in-box; OpenAI and Voyage adapters slot in as new modules with a couple lines each. See [`providers/anthropic.py`](src/llm_cost_governor/providers/anthropic.py) for the shape.

### State backends

Counters can persist their rolling-window state through the `StateBackend` Protocol. Two implementations ship:

- `LocalFileBackend(path)` — atomic JSON writes to a filesystem path. Default for local dev / CI.
- `GcsBackend(bucket)` — Google Cloud Storage blob. Lazily imports `google-cloud-storage` on first use, so the core install stays dep-free.

Add your own by implementing `read(name) -> str | None` and `write(name, text) -> None`.

### `record_usage` — for calls you made yourself

Voyage embeddings, batch APIs, vision — anything that doesn't fit the `guarded_call` shape. `record_usage()` runs only the **post** hooks, still gives you priced cost and event log, without wrapping the call:

```python
from llm_cost_governor.wrapper import record_usage

response = voyage_client.embed(texts=[...], model="voyage-3.5")
record_usage(
    provider="voyage", model="voyage-3.5",
    input_tokens=response.total_tokens, output_tokens=0,
    hooks=hooks, tags={"call_type": "embedding"},
)
```

---

## What's in / what's out

**Included:**
- Pricing for currently-shipped Claude models — Fable 5, Opus 5, Sonnet 5, Opus 4.6/4.7/4.8, Sonnet 4.6, Haiku 4.5. Easy to extend for new models as they ship.
- Rolling-window counter with configurable caps + durable persistence.
- `RollingWeekCounter` — a generic per-key, rolling-week cumulative counter with cap enforcement (strict or lenient) across one or more named dimensions; the reusable core behind app-specific caps like per-token/per-IP upload limits. Import from `llm_cost_governor.counters`.
- `ProviderTotals` + `ProviderTotalsHook` — a per-provider cumulative-USD read-model (anthropic vs voyage vs …) for usage widgets and cost dashboards; the breakdown the windowed `CostCounter` aggregates away. In-memory by default, optionally persisted through a `StateBackend`. Import from `llm_cost_governor.provider_totals`.
- Per-IP rate limiter (framework-neutral core + FastAPI dependency factory).
- Structured event log (stdout → any log aggregator).
- Provider adapter for Anthropic.
- OTel span hooks + a request-span context manager + LangSmith metadata.
- Content-scrubbing OTel exporter for per-request telemetry control.
- Discord-webhook alert sink (implements the `AlertSink` Protocol).

**Not (yet) included:**
- Provider adapters for OpenAI, Voyage, Gemini — the shape is fixed and each is ~30 lines, but they're not in the box until someone needs them.
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

Currently `v0.4.4`. The `0.3.x` line renamed the package from `llm-guardrails` to `llm-cost-governor`; `0.4.0` adds `capability` to every model record; `0.4.1` adds `RequirePricedModelHook`, the pre-flight gate that stops an unpriced model from silently escaping every budget and cap, plus OpenAI GPT-5 and embedding rates; `0.4.2` meters Anthropic server-side tool use (web search) instead of dropping it; `0.4.3` adds the OpenAI provider adapter and real OpenAI cache rates; `0.4.4` adds `provider` to every model record. Semver from `v1.0.0` onward; anything before is "shipped but pre-stable API — expect breaking changes."

## Contributing

Issues and pull requests welcome. For substantive changes, open an issue first to discuss the shape before writing code. The Hook Protocol and provider-adapter surface are the two most important extension points — happy to talk through how to add a new provider or hook.

## License

MIT. See [LICENSE](LICENSE).

---

_Last updated:_ 2026-08-27
