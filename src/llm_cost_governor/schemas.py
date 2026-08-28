# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Wire-shape data classes for the llm-cost-governor library.

Two lightweight pydantic models the wrapper + hooks share:

  * `UsageRecord` — the normalized per-call cost + token record every
    hook sees in its post step, and that `guarded_call` / `record_usage`
    return to the caller. Provider-agnostic: cache-token fields are
    ``int | None`` so a provider that doesn't expose them (OpenAI,
    Voyage) can leave them absent rather than lie with a zero.

  * `TokenEstimate` — the pre-flight token estimate that budget/cap
    hooks use to decide whether to allow the call. Provider adapters
    compute a default from the caller's kwargs; callers can override.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field


class TokenEstimate(BaseModel):
    """Pre-flight token estimate for a call.

    Consumed by budget/cap hooks in their ``pre`` step to decide
    whether the call would push the caller over a limit. The provider
    adapter's ``default_estimate`` supplies a conservative value from
    the outbound ``kwargs``; callers may override.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int


class UsageRecord(BaseModel):
    """One completed LLM call's normalized usage + cost + tags.

    Returned by `guarded_call` and `record_usage`, and passed to every
    hook's ``post`` step. Cache-token fields are ``int | None``:
    ``None`` means "the provider doesn't expose this metric" (e.g.
    OpenAI, Voyage), while ``0`` means "the provider reported zero" —
    distinguishing them matters for accurate accounting downstream.

    ``server_tool_use`` carries the provider's non-token billing counts
    for the call (e.g. ``{"web_search_requests": 2}``) — ``None`` when the
    call used no server-side tools. It is reported separately from the
    token fields because it is billed in a different unit; ``cost_usd``
    already includes its priced portion.

    The ``tags`` dict is the caller's opaque annotation space (stage,
    agent, route, qa_id, whatever). Hooks that need to key on caller
    context read from ``tags`` rather than requiring the library to
    know the shape.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cost_usd: float
    server_tool_use: dict[str, int] | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)
