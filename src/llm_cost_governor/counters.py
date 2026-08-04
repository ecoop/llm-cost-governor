# Copyright (c) 2026 Eric Cooper. Licensed under MIT; see LICENSE.
"""Rolling-window LLM spend counter with cap enforcement.

The hot-path enforcement layer for LLM spend. A ``CostCounter``
instance tracks spend in four rolling windows and blocks new calls
when any cap is tripped:

    - Global hourly (rolling UTC hour)
    - Global daily  (rolling UTC day)
    - Global weekly (rolling UTC week)
    - Per-identity  (rolling weekly, when a caller provides an
                    identity string on record()/enforce())

Zero coupling to any host application: every value the counter needs
(caps, enforcement gate, backend, object name for persistence) is a
constructor arg. Callers configure at instantiation and hold their
own instance.

Lifecycle:
    - ``counter.load()`` runs once at app startup. Reads the state
      blob into memory and starts a background writer thread when
      ``enabled`` was true at construction; when off, just initializes
      empty in-memory state.
    - ``counter.enforce(identity)`` runs pre-flight; no-op unless
      ``enabled`` is true. Over a cap → raises ``CostCapExceeded``.
    - ``counter.record(...)`` runs post-response; increments the
      in-memory windows, schedules a coalesced blob write, and emits
      operator alerts at cap-trip points via the
      ``llm_cost_governor.alerts`` seam.

State model:
    In-memory dicts are the source of truth between writes. The state
    blob is a durable snapshot rewritten fire-and-forget after
    responses. A crash between writes loses at most the last few cents
    of recorded spend. Missing blob on startup → start at zero
    (conservative: caps still bind immediately).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from llm_cost_governor.alerts import CRITICAL, WARNING, alert
from llm_cost_governor.persistence import GcsBackedCounter
from llm_cost_governor.pricing import usd_for_usage
from llm_cost_governor.schemas import UsageRecord
from llm_cost_governor.state import StateBackend
from llm_cost_governor.wrapper import CallContext

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Retention: keep a bounded history so the blob stays small.
_MAX_HOURLY = 25
_MAX_DAILY = 7
_MAX_WEEKLY = 4


# ── Exception ─────────────────────────────────────────────────────────────────


class CostCapExceeded(Exception):
    """Raised pre-flight when a spend cap is already at/over its limit.

    Attributes:
        cap: Machine-readable cap id — ``"hourly"``, ``"daily"``,
            ``"weekly"``, or ``"per_token"``. Lets the API layer /
            alerting branch on which cap tripped without parsing the
            message.

    The message is user-facing; api.main renders it as an HTTP 402 body.
    """

    def __init__(self, cap: str, message: str) -> None:
        super().__init__(message)
        self.cap = cap


# ── Time-bucket helpers (UTC) ──────────────────────────────────────────────────


def _this_hour() -> str:
    """Current UTC hour bucket as ``YYYY-MM-DDTHH``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H")


def _today() -> str:
    """Current UTC date as ``YYYY-MM-DD``."""
    return datetime.now(UTC).date().isoformat()


def _week_of(d: date | None = None) -> str:
    """Monday (ISO week start) of ``d`` (default today, UTC) as ``YYYY-MM-DD``."""
    d = d or datetime.now(UTC).date()
    return (d.fromordinal(d.toordinal() - d.weekday())).isoformat()


def _now_iso() -> str:
    """Current UTC timestamp, second precision, as an ISO string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# ── Per-token accounting record ────────────────────────────────────────────────


@dataclass
class _TokenStat:
    """Rolling-week spend for one identity (e.g. an invite token).

    ``usd_cumulative_week`` resets to zero when ``week_of`` rolls
    over, so the per-identity cap is a weekly allowance rather than a
    lifetime one.
    """

    usd_cumulative_week: float = 0.0
    week_of: str = field(default_factory=_week_of)
    first_seen: str = field(default_factory=_now_iso)
    last_seen: str = field(default_factory=_now_iso)
    call_count: int = 0


# ── Counter ─────────────────────────────────────────────────────────────────────


class CostCounter(GcsBackedCounter):
    """In-memory rolling-window spend tracker with coalesced persistence.

    The lock, persistence lifecycle (load/shutdown/flush), and
    background writer thread come from ``GcsBackedCounter``. This
    subclass adds the spend-specific state — four rolling windows
    plus per-identity weekly accounting — and the enforce / record /
    snapshot logic.

    Args:
        object_name: Filename/key for the persisted state blob.
        backend: Where to persist state. Any ``StateBackend`` (local,
            GCS, or a caller-supplied in-memory fake for tests).
        enabled: Master gate for enforcement + persistence. When
            False, the counter still records in memory (local
            visibility) but ``enforce()`` is a no-op and no writes
            happen.
        hourly_cap_usd / daily_cap_usd / weekly_cap_usd /
            per_token_cap_usd: The four rolling-window ceilings.
            Enforcement raises ``CostCapExceeded`` when a window is
            at or over its cap.
    """

    def __init__(
        self,
        *,
        object_name: str,
        backend: StateBackend,
        enabled: bool = False,
        hourly_cap_usd: float = 0.50,
        daily_cap_usd: float = 2.00,
        weekly_cap_usd: float = 10.00,
        per_token_cap_usd: float = 1.00,
    ) -> None:
        super().__init__(
            name="cost_counter",
            object_name=object_name,
            writer_thread_name="cost-counter-writer",
            log=_log,
        )
        self._backend = backend
        self._enabled = enabled
        self._hourly_cap_usd = hourly_cap_usd
        self._daily_cap_usd = daily_cap_usd
        self._weekly_cap_usd = weekly_cap_usd
        self._per_token_cap_usd = per_token_cap_usd

        self._hourly: dict[str, float] = {}       # hour bucket → usd
        self._daily: dict[str, float] = {}        # date  → usd
        self._weekly: dict[str, float] = {}       # week_of → usd
        self._per_token: dict[str, _TokenStat] = {}
        self._alerted: set[str] = set()           # per-window latch (#275 gap 3)

    def load(self, *, backend_kind: str = "") -> None:  # type: ignore[override]
        """Initialize state; read backend + start writer when ``enabled``.

        Forwards to :meth:`GcsBackedCounter.load` with the counter's
        own ``enabled`` gate. Kept as a convenience so consumers only
        pass ``enabled`` once (at construction) rather than repeating
        it at load time.
        """
        super().load(enabled=self._enabled, backend_kind=backend_kind)

    # ── enforcement ──────────────────────────────────────────────────────────────

    def enforce(self, token: str | None) -> None:
        """Pre-flight cap check. No-op unless ``enabled`` is on.

        Compares the *already-recorded* totals against each cap. A
        call that crosses a cap is allowed to complete; the next one
        is blocked — a one-call overshoot we accept rather than trying
        to predict the in-flight call's cost (the pre-flight estimate
        is unreliable for cache-heavy prompts).

        Args:
            token: The caller's identity (invite token, user id, or
                whatever the app uses), or None (no per-identity check).

        Raises:
            CostCapExceeded: When an hourly, daily, weekly, or
                per-token cap is met. Checked tightest-window-first
                so the message names the cap with the soonest reset.
        """
        if not self._enabled:
            return

        with self._lock:
            hourly = self._hourly.get(_this_hour(), 0.0)
            daily = self._daily.get(_today(), 0.0)
            weekly = self._weekly.get(_week_of(), 0.0)
            per_token = 0.0
            if token is not None:
                stat = self._per_token.get(token)
                if stat is not None and stat.week_of == _week_of():
                    per_token = stat.usd_cumulative_week

        if hourly >= self._hourly_cap_usd:
            raise CostCapExceeded(
                "hourly",
                f"The demo's hourly spend cap (${self._hourly_cap_usd:.2f}) has been "
                f"reached. It resets each hour — please try again shortly.",
            )
        if daily >= self._daily_cap_usd:
            raise CostCapExceeded(
                "daily",
                f"The demo's daily spend cap (${self._daily_cap_usd:.2f}) has been "
                f"reached. It resets at UTC midnight — please try again tomorrow.",
            )
        if weekly >= self._weekly_cap_usd:
            raise CostCapExceeded(
                "weekly",
                f"The demo's weekly spend cap (${self._weekly_cap_usd:.2f}) has been "
                f"reached. Please check back next week.",
            )
        if token is not None and per_token >= self._per_token_cap_usd:
            raise CostCapExceeded(
                "per_token",
                f"Your invite's weekly spend allowance (${self._per_token_cap_usd:.2f}) "
                f"has been reached. It resets weekly.",
            )

    # ── recording ──────────────────────────────────────────────────────────────

    def record(self, usage: dict, token: str | None = None) -> float:
        """Price a completed call and add it to every rolling window.

        Runs in all modes (local visibility); only schedules a persist
        write when persisting. Per-token tracking is skipped when
        ``token`` is None (e.g. offline calls that run outside a
        request scope).

        Args:
            usage: Anthropic ``usage``-shaped dict (see
                llm_cost_governor.pricing.usd_for_usage).
            token: The caller's identity, or None.

        Returns:
            The USD cost attributed to this call.
        """
        usd = usd_for_usage(usage)
        hour = _this_hour()
        today = _today()
        week = _week_of()
        now = _now_iso()

        with self._lock:
            self._hourly[hour] = self._hourly.get(hour, 0.0) + usd
            self._daily[today] = self._daily.get(today, 0.0) + usd
            self._weekly[week] = self._weekly.get(week, 0.0) + usd
            self._trim()

            if token is not None:
                stat = self._per_token.get(token)
                if stat is None or stat.week_of != week:
                    # New token, or its weekly allowance has rolled over.
                    stat = _TokenStat(week_of=week, first_seen=now)
                    self._per_token[token] = stat
                stat.usd_cumulative_week += usd
                stat.last_seen = now
                stat.call_count += 1

            # Decide which alerts to fire *under the lock* (reads the just-
            # updated totals and flips the per-window latch atomically), but
            # deliver them after releasing it so a slow/blocking delivery
            # never serializes recording threads.
            pending = self._collect_alerts(hour, today, week, token)

        for level, title, body in pending:
            # llm_cost_governor.alerts.alert() is best-effort by contract (it
            # catches and logs sink failures), but we keep a belt-and-
            # suspenders try/except so a misbehaving stand-in patched into
            # this module's namespace can't interrupt the batch either.
            try:
                alert(level, title, body)
            except Exception:  # noqa: BLE001 — an alert must never break a response
                _log.exception("cost_counter: alert delivery failed.")

        self._schedule_write()
        return usd

    def _collect_alerts(
        self, hour: str, today: str, week: str, token: str | None
    ) -> list[tuple[str, str, str]]:
        """Pick cap-trip alerts to emit, latching each once per window.

        Caller holds the lock. Returns ``(level, title, body)`` tuples
        for `alert`; the actual delivery happens after the lock is
        released. A no-op unless ``enabled`` is on.

        Latching: each alert keys the relevant window id into
        ``self._alerted`` so it fires once per window. When the window
        rolls over the key changes, so the alert re-arms for the new
        window. Trip points and severities: daily reached → 🔴;
        weekly at 80%, hourly reached, and per-token reached → 🟡.
        """
        if not self._enabled:
            return []

        out: list[tuple[str, str, str]] = []

        daily = self._daily.get(today, 0.0)
        if daily >= self._daily_cap_usd and self._latch(f"daily:{today}"):
            out.append((
                CRITICAL,
                "Demo daily spend cap reached",
                (
                    f"Daily spend ${daily:.2f} reached the ${self._daily_cap_usd:.2f} "
                    f"cap — new calls are blocked until UTC midnight."
                ),
            ))

        weekly = self._weekly.get(week, 0.0)
        if weekly >= 0.8 * self._weekly_cap_usd and self._latch(f"weekly80:{week}"):
            out.append((
                WARNING,
                "Demo weekly spend at 80% of cap",
                (
                    f"Weekly spend ${weekly:.2f} crossed 80% of the "
                    f"${self._weekly_cap_usd:.2f} weekly cap."
                ),
            ))

        hourly = self._hourly.get(hour, 0.0)
        if hourly >= self._hourly_cap_usd and self._latch(f"hourly:{hour}"):
            out.append((
                WARNING,
                "Demo hourly spend cap reached",
                (
                    f"Hourly spend ${hourly:.2f} reached the ${self._hourly_cap_usd:.2f} "
                    f"cap — resets at the top of the next hour."
                ),
            ))

        if token is not None:
            stat = self._per_token.get(token)
            if (
                stat is not None
                and stat.week_of == week
                and stat.usd_cumulative_week >= self._per_token_cap_usd
                and self._latch(f"per_token:{token}:{week}")
            ):
                out.append((
                    WARNING,
                    "Invite token weekly allowance reached",
                    (
                        f"An invite token's weekly spend ${stat.usd_cumulative_week:.2f} "
                        f"reached the ${self._per_token_cap_usd:.2f} per-token cap."
                    ),
                ))

        return out

    def _latch(self, key: str) -> bool:
        """Return True the first time ``key`` is seen, then False. Holds lock."""
        if key in self._alerted:
            return False
        self._alerted.add(key)
        return True

    # ── snapshots / serialization ──────────────────────────────────────────────

    def current_usage(self) -> dict:
        """Current-window totals for the diagnostics panel.

        Returns a small dict: today's spend, this week's spend, and
        the configured caps, plus the per-token weekly breakdown.
        """
        with self._lock:
            return {
                "hourly_usd": round(self._hourly.get(_this_hour(), 0.0), 4),
                "daily_usd": round(self._daily.get(_today(), 0.0), 4),
                "weekly_usd": round(self._weekly.get(_week_of(), 0.0), 4),
                "caps": {
                    "hourly_usd": self._hourly_cap_usd,
                    "daily_usd": self._daily_cap_usd,
                    "weekly_usd": self._weekly_cap_usd,
                    "per_token_usd": self._per_token_cap_usd,
                },
                "per_token": {
                    tok: round(s.usd_cumulative_week, 4)
                    for tok, s in self._per_token.items()
                    if s.week_of == _week_of()
                },
            }

    def snapshot_for(self, token: str | None) -> dict:
        """Tester-facing usage snapshot: global windows + caps + caller's line.

        The privacy boundary for the demo-usage panel. The full
        ``per_token`` map is keyed by raw invite tokens — credentials
        — so it must never reach the browser. This returns the global
        daily / weekly spend, the configured spend caps, and ONLY the
        given token's own current-week cumulative (or ``None`` when
        no/unknown token).

        Args:
            token: The caller's invite token, or None (no caller line).

        Returns:
            ``{hourly_usd, daily_usd, weekly_usd, caps{hourly_usd,
            daily_usd, weekly_usd, per_token_usd}, caller_weekly_usd}``
            — all rounded; no other token's data included.
        """
        with self._lock:
            hourly = self._hourly.get(_this_hour(), 0.0)
            daily = self._daily.get(_today(), 0.0)
            weekly = self._weekly.get(_week_of(), 0.0)
            caller_weekly: float | None = None
            if token is not None:
                stat = self._per_token.get(token)
                if stat is not None and stat.week_of == _week_of():
                    caller_weekly = round(stat.usd_cumulative_week, 4)
        return {
            "hourly_usd": round(hourly, 4),
            "daily_usd": round(daily, 4),
            "weekly_usd": round(weekly, 4),
            "caps": {
                "hourly_usd": self._hourly_cap_usd,
                "daily_usd": self._daily_cap_usd,
                "weekly_usd": self._weekly_cap_usd,
                "per_token_usd": self._per_token_cap_usd,
            },
            "caller_weekly_usd": caller_weekly,
        }

    def to_dict(self) -> dict:
        """Serialize current state to the persisted-blob schema."""
        with self._lock:
            hourly = sorted(self._hourly.items())[-_MAX_HOURLY:]
            daily = sorted(self._daily.items())[-_MAX_DAILY:]
            weekly = sorted(self._weekly.items())[-_MAX_WEEKLY:]
            return {
                "schema_version": SCHEMA_VERSION,
                "hourly": [{"hour": h, "usd": round(u, 6)} for h, u in hourly],
                "daily": [{"date": d, "usd": round(u, 6)} for d, u in daily],
                "weekly": [{"week_of": w, "usd": round(u, 6)} for w, u in weekly],
                "per_token": {
                    tok: {
                        "usd_cumulative_week": round(s.usd_cumulative_week, 6),
                        "week_of": s.week_of,
                        "first_seen": s.first_seen,
                        "last_seen": s.last_seen,
                        "call_count": s.call_count,
                    }
                    for tok, s in self._per_token.items()
                },
            }

    def from_dict(self, data: dict) -> None:
        """Replace in-memory state from a deserialized blob (tolerant of gaps)."""
        with self._lock:
            self._hourly = {
                row["hour"]: float(row["usd"])
                for row in data.get("hourly", [])
                if "hour" in row
            }
            self._daily = {
                row["date"]: float(row["usd"])
                for row in data.get("daily", [])
                if "date" in row
            }
            self._weekly = {
                row["week_of"]: float(row["usd"])
                for row in data.get("weekly", [])
                if "week_of" in row
            }
            self._per_token = {}
            for tok, s in data.get("per_token", {}).items():
                self._per_token[tok] = _TokenStat(
                    usd_cumulative_week=float(s.get("usd_cumulative_week", 0.0)),
                    week_of=s.get("week_of", _week_of()),
                    first_seen=s.get("first_seen", _now_iso()),
                    last_seen=s.get("last_seen", _now_iso()),
                    call_count=int(s.get("call_count", 0)),
                )
            self._trim()

    # ── internals ──────────────────────────────────────────────────────────────

    def _trim(self) -> None:
        """Bound history to the retention windows. Caller holds the lock."""
        if len(self._hourly) > _MAX_HOURLY:
            for k in sorted(self._hourly)[:-_MAX_HOURLY]:
                del self._hourly[k]
        if len(self._daily) > _MAX_DAILY:
            for k in sorted(self._daily)[:-_MAX_DAILY]:
                del self._daily[k]
        if len(self._weekly) > _MAX_WEEKLY:
            for k in sorted(self._weekly)[:-_MAX_WEEKLY]:
                del self._weekly[k]

    # ── persistence hooks (GcsBackedCounter) ───────────────────────────────────

    def _read_blob(self) -> str | None:
        return self._backend.read(self._object_name)

    def _write_blob(self, text: str) -> None:
        self._backend.write(self._object_name, text)

    def _write_failure_alert(self) -> tuple[str, str, str]:
        """Operator alert when persisting spend fails.

        The 5-min dedupe in alerts keeps a persistent outage from
        flooding the channel on every record.
        """
        return (
            WARNING,
            "Cost-counter GCS write failed",
            (
                "Persisting demo spend to the state blob failed; in-memory "
                "totals are intact but won't survive a restart until a "
                "write succeeds. Check the GCS bucket / IAM."
            ),
        )


# ── Hook adapter for guarded_call ──────────────────────────────────────────────


class WindowedCapHook:
    """`Hook` adapter that enforces a `CostCounter`'s caps in a `guarded_call`.

    `pre` calls the counter's pre-flight `enforce` (raises
    `CostCapExceeded` when a rolling-window cap is at or over its
    limit). `post` records the actual call cost from the returned
    `UsageRecord`, which may trip cap-reached alerts.

    Identity resolution is delegated to a caller-supplied callable
    (returning a token / user id / ``None``) so the same hook serves
    both authenticated apps (pitchcraft's invite tokens) and anonymous
    ones (rulebook, which passes `identity_provider=None` — the
    default — and gets only global-window enforcement).
    """

    name = "windowed_cap"

    def __init__(
        self,
        counter: CostCounter,
        identity_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self.counter = counter
        self.identity_provider = identity_provider or (lambda: None)

    def pre(self, ctx: CallContext) -> None:
        self.counter.enforce(token=self.identity_provider())

    def post(self, ctx: CallContext, usage: UsageRecord) -> None:
        self.counter.record(
            {
                "model": usage.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            },
            token=self.identity_provider(),
        )
