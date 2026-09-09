# Integration Guide

How to adopt `llm-cost-governor` in a Python application. This is the practical companion to the [README](../README.md) — the README explains *what* the library does; this doc explains *how* to wire it into your app.

**Reference implementation:** [Rulebook](https://github.com/ecoop/rulebook) consumes this library in production and is public, so every file referenced below is readable. Its [`src/rulebook/app_state.py`](https://github.com/ecoop/rulebook/blob/main/src/rulebook/app_state.py) is the canonical adoption pattern.

---

## Install

Core install:

```bash
pip install llm-cost-governor
```

With optional integrations:

```bash
pip install "llm-cost-governor[gcs,fastapi,otel]"
```

The optional extras:

- **`[gcs]`** — `GcsBackend` for durable counter state in production
- **`[fastapi]`** — Request-scoped helpers (IP rate-limit dependency factory)
- **`[otel]`** — OTel span hooks + LangSmith metadata

Core install has just one dependency (`pydantic>=2.0`). Every integration is behind an optional extra so the base install stays lean.

---

## The dependency-injection pattern

The library **takes zero configuration from the host app.** Every class takes explicit constructor arguments; there are no module singletons inside the library, no config imports, no environment-variable reads.

The host app is responsible for:

1. Reading its own config
2. Constructing the library's singletons at startup
3. Exposing them through a facade module that consumers import

Rulebook's [`app_state.py`](https://github.com/ecoop/rulebook/blob/main/src/rulebook/app_state.py) is the working example. The shape:

```python
# app_state.py — the singleton facade
from pathlib import Path
from typing import Callable, Optional
from fastapi import Request

from llm_cost_governor.counters import CostCounter
from llm_cost_governor.ratelimit import IPRateLimiter
from llm_cost_governor.state import get_backend, StateBackend
from llm_cost_governor.fastapi_ext import make_enforce_ip_rate_limit

cost_counter: Optional[CostCounter] = None
ip_rate_limiter: Optional[IPRateLimiter] = None
state_backend: Optional[StateBackend] = None
_enforce_ip_rate_limit_impl: Optional[Callable[[Request], None]] = None


def initialize(settings, *, data_dir: Path) -> None:
    """Called from main.py at startup, before any FastAPI router is imported."""
    global cost_counter, ip_rate_limiter, state_backend, _enforce_ip_rate_limit_impl

    state_backend = get_backend(
        kind=settings.state_backend_kind,
        data_dir=data_dir,
        gcs_bucket=settings.gcs_state_bucket,
    )
    cost_counter = CostCounter(
        object_name="cost_counter.json",
        backend=state_backend,
        enabled=settings.demo_mode,
        hourly_cap_usd=settings.cap_hourly_usd,
        daily_cap_usd=settings.cap_daily_usd,
        weekly_cap_usd=settings.cap_weekly_usd,
        per_token_cap_usd=settings.cap_per_token_usd,
    )
    cost_counter.load()

    ip_rate_limiter = IPRateLimiter()
    _enforce_ip_rate_limit_impl = make_enforce_ip_rate_limit(
        ip_rate_limiter,
        cap_rpm=settings.rate_limit_rpm,
        enabled=settings.demo_mode,
    )


def enforce_ip_rate_limit(request: Request) -> None:
    """Stable module-level wrapper — safe to pass to Depends() at import time."""
    if _enforce_ip_rate_limit_impl is None:
        raise RuntimeError("app_state.enforce_ip_rate_limit invoked before initialize()")
    _enforce_ip_rate_limit_impl(request)
```

Call `initialize()` once at startup. Consumers then import the module-level names (`app_state.cost_counter`, `app_state.enforce_ip_rate_limit`) rather than the library classes directly.

---

## FastAPI: initialization-order gotcha

If your app uses FastAPI, watch for this: **FastAPI routers call `Depends(app_state.enforce_ip_rate_limit)` at import time, before `app_state.initialize()` runs.** That's why the pattern above defines `enforce_ip_rate_limit` as a stable module-level function that internally reads a lazily-set `_enforce_ip_rate_limit_impl` variable — not as a name that gets assigned inside `initialize()`.

The naive version breaks:

```python
# ❌ WRONG — module-level assignment happens too late
def initialize(settings, *, data_dir):
    global enforce_ip_rate_limit
    enforce_ip_rate_limit = make_enforce_ip_rate_limit(...)
```

Routers importing `app_state.enforce_ip_rate_limit` at module load time capture the reference **before** `initialize()` runs, so they bind to `None` (or whatever the module-level default is). No amount of runtime setup fixes it — `Depends()` has already snapshot the wrong value.

The fix is what the working example shows: a stable wrapper function that reads through a lazily-set implementation variable.

Similarly, in `main.py`, call `app_state.initialize()` **before** importing the FastAPI app or any of its routers:

```python
# main.py — order matters
from config import DATA_DIR, settings

import app_state
app_state.initialize(settings, data_dir=DATA_DIR)  # BEFORE the next line

from api.main import app  # noqa: E402 — routers get imported here
```

---

## Constructor signatures

All shipped classes take keyword-only arguments and don't read config on their own.

### `CostCounter`

```python
CostCounter(
    *,
    object_name: str,             # e.g. "cost_counter.json"
    backend: StateBackend,        # from get_backend()
    enabled: bool = False,        # gates all enforcement + persistence
    hourly_cap_usd: float = 0.50,
    daily_cap_usd: float = 2.00,
    weekly_cap_usd: float = 10.00,
    per_token_cap_usd: float = 1.00,
)
```

Call `.load()` after construction to restore persisted state from the backend.

### `get_backend` (state factory)

```python
get_backend(
    kind: str,                     # "local" or "gcs"
    *,
    data_dir: Path | None = None,  # required if kind="local"
    gcs_bucket: str | None = None, # required if kind="gcs"
) -> StateBackend
```

### `IPRateLimiter` + `make_enforce_ip_rate_limit`

```python
limiter = IPRateLimiter()  # no args

enforce = make_enforce_ip_rate_limit(
    limiter,
    *,
    cap_rpm: int,
    enabled: bool = False,
) -> Callable[[Request], None]
```

The returned callable is a FastAPI dependency (safe to pass to `Depends()`).

### `EventLogHook` + `emit_llm_call`

```python
EventLogHook(
    *,
    enabled: bool = False,
    identity_provider: Callable[[], Mapping | None] = lambda: None,
    session_id_provider: Callable[[], str | None] = lambda: None,
)

emit_llm_call(
    *,
    enabled: bool = False,
    stage: str,
    agent: str,
    usage: Mapping,           # Anthropic-shaped usage dict
    usd: float,
    identity: Mapping | None = None,
    session_id: str | None = None,
)
```

`identity_provider` and `session_id_provider` are callables so the hook can read the current request-scoped identity + session ID at post-call time (typically via `ContextVars`). Pitchcraft uses this to stamp the invited user's token hash and the ULID session ID on every logged call.

---

## Shipped hooks

Every hook satisfies the `Hook` Protocol (`pre(ctx)` + `post(ctx, usage)` methods). Compose them into a list and pass as the `hooks=` argument to `guarded_call()`.

| Hook | Module | Purpose |
|---|---|---|
| `RequirePricedModelHook` | `llm_cost_governor.budget` | Pre-flight refusal of a model the pricing table can't cost |
| `ScopeBudgetHook` | `llm_cost_governor.budget` | Pre-flight budget enforcement per session/scope |
| `WindowedCapHook` | `llm_cost_governor.counters` | Rolling-window cost cap enforcement + alert |
| `EventLogHook` | `llm_cost_governor.events` | Structured JSON event per call (stdout) |
| `ProviderTotalsHook` | `llm_cost_governor.provider_totals` | Per-provider cumulative USD, for usage widgets |
| `OTelSpanHook` | `llm_cost_governor.otel.hooks` | Per-call OTel span with cost + token attrs |
| `LangSmithMetadataHook` | `llm_cost_governor.otel.hooks` | Stamps LangSmith metadata from identity |

Typical chain:

```python
hooks = [
    RequirePricedModelHook(),        # refuse anything the table can't price
    ScopeBudgetHook(session_budget),
    WindowedCapHook(app_state.cost_counter),
    EventLogHook(enabled=settings.event_log_enabled, ...),
    OTelSpanHook(),
    LangSmithMetadataHook(identity_provider=get_invite_identity),
]

response, usage = guarded_call(
    client, provider="anthropic", hooks=hooks,
    tags={"stage": "drafter"}, model="claude-sonnet-5",
    messages=[...], max_tokens=1024,
)
```

For non-`guarded_call` shapes (Voyage embeddings, batch APIs), use `record_usage()` — same hook chain, no wrapper.

### A gated ceiling in one call

If all you need is a hard USD ceiling that can't be silently escaped, skip the hand-assembly:

```python
from llm_cost_governor.budget import build_budget_chain

hooks = build_budget_chain(limit_usd=3.00)   # [RequirePricedModelHook, ScopeBudgetHook]
```

One import that either resolves or raises `ImportError` — which is what stops a
too-old install from degrading silently to no enforcement. Pass `budget=` instead of
`limit_usd=` to share one ceiling across many calls and read `spent_usd` afterwards.

---

## Reference implementation: Rulebook

[Rulebook](https://github.com/ecoop/rulebook) is public and exercises most of the library's
surface — persistence, per-identity caps, rate limiting, and the per-provider read-model.
Files worth skimming, in priority order:

1. [`src/rulebook/app_state.py`](https://github.com/ecoop/rulebook/blob/main/src/rulebook/app_state.py) — the singleton facade (the pattern to copy)
2. [`api/main.py`](https://github.com/ecoop/rulebook/blob/main/api/main.py) — where initialization happens relative to router import
3. [`src/rulebook/generate.py`](https://github.com/ecoop/rulebook/blob/main/src/rulebook/generate.py) — hook-chain composition around `guarded_call`
4. [`src/rulebook/embeddings.py`](https://github.com/ecoop/rulebook/blob/main/src/rulebook/embeddings.py) — the `record_usage` path for a provider with no adapter

### A pattern worth stealing: fail closed at boot

`app_state.py` refuses to start when a configured model can't be priced:

> *"a model llm-cost-governor can't price bills $0, so it contributes nothing to the
> rolling cost windows and `WindowedCapHook` never trips for it — the caps silently
> under-enforce. Refuse to boot when guardrails are on and any configured model prices
> at zero."*

`RequirePricedModelHook` catches this per call; the boot check catches a misconfiguration
before any traffic arrives, and turns a silent under-enforcement into a startup failure.
It degrades to a warning when guardrails are off, and skips rather than crashing if the
pricing symbol ever moves.

---

## Known gaps

### Some providers are priced but have no adapter

`MODEL_PRICING` covers 38 models — Claude, the OpenAI GPT-5 family and embeddings, and
Voyage embeddings and rerank — each row tagged with its `provider` and `capability`.
Adapters, which is what `guarded_call` needs, ship for **Anthropic and OpenAI** only;
`providers.ADAPTERS` is the authoritative list.

That asymmetry is deliberate, not an oversight: a rate row means "we can cost this", an
adapter means "we can also wrap the call and run pre-flight hooks". Voyage models are
priced and metered through `record_usage()`, which needs no adapter.

For a model the table doesn't know at all, `usd_for_usage()` returns `$0` and fires a
one-time alert through the `AlertSink` protocol — and `RequirePricedModelHook` refuses
the call pre-flight, which is the behaviour you want, since a `$0` model is exempt from
every budget and cap rather than merely cheap. To add one, open a small PR against
[`src/llm_cost_governor/pricing.py`](../src/llm_cost_governor/pricing.py); everyone
benefits, and the table is the only place rates live.

### Not yet included

- Streaming responses — the wrapper is synchronous today.
- Multi-instance atomic counters — the rolling-window counter is correct at
  `max-instances=1`. Under horizontal scaling each instance holds its own counter, so a
  shared cap is enforced independently per instance; distributed correctness (e.g.
  Redis-backed) is a future extension.
- Retry / circuit-breaker logic — the library never retries; that's the caller's job.

---

## Getting help

For questions on the adoption pattern that aren't covered here: open an issue on this repo, or read the Rulebook files above — they're the working ground truth.

---

_Last updated:_ 2026-09-08
