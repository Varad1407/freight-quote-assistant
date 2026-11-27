"""In-memory rate limiting for the public demo.

Why: the demo calls a free LLM API on my account. Without limits, one visitor (or a bot)
could use up the daily quota and the demo would stop working for everyone else.

Two limits:
1. Per visitor (IP address): at most `per_ip_limit` requests per `window_s` seconds (sliding window).
2. Global: at most `global_daily_limit` requests per rolling 24 hours, a safety net for the quota.

Limitations (fine for a one-instance demo, not for a fleet):
- State lives in memory, so it resets on restart and is not shared between instances.
  A multi-instance deployment would keep these counters in Redis instead.
"""

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

DAY_S = 24 * 60 * 60


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after_s: int = 0
    reason: str = ""


class RateLimiter:
    def __init__(
        self,
        per_ip_limit: int,
        window_s: float,
        global_daily_limit: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._per_ip_limit = per_ip_limit
        self._window_s = window_s
        self._global_limit = global_daily_limit
        self._clock = clock  # injectable so tests don't have to sleep
        self._hits: dict[str, deque[float]] = {}
        self._global_hits: deque[float] = deque()
        self._lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool

    def check(self, key: str) -> Decision:
        """Record one request for `key` if it is allowed. Rejected requests are not counted."""
        now = self._clock()
        with self._lock:
            _drop_older_than(self._global_hits, now - DAY_S)
            if 0 < self._global_limit <= len(self._global_hits):
                wait = self._global_hits[0] + DAY_S - now
                return Decision(False, _ceil(wait), "global")

            hits = self._hits.setdefault(key, deque())
            _drop_older_than(hits, now - self._window_s)
            if 0 < self._per_ip_limit <= len(hits):
                wait = hits[0] + self._window_s - now
                return Decision(False, _ceil(wait), "per_ip")

            hits.append(now)
            self._global_hits.append(now)
            if len(self._hits) > 10_000:  # keep memory bounded if many different IPs visit
                self._forget_idle(now)
            return Decision(True)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._global_hits.clear()

    def _forget_idle(self, now: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] < now - self._window_s]:
            del self._hits[key]


def client_ip(forwarded_for: str | None, direct_ip: str | None) -> str:
    """The visitor's IP address.

    Behind Render's proxy, every request arrives from the proxy, so the real address is in
    X-Forwarded-For. Each proxy APPENDS the address it saw, so the last entry is the one
    added by Render; earlier entries can be typed in by the visitor to dodge the limit.
    """
    if forwarded_for:
        last = forwarded_for.split(",")[-1].strip()
        if last:
            return last
    return direct_ip or "unknown"


def _drop_older_than(hits: deque[float], cutoff: float) -> None:
    while hits and hits[0] <= cutoff:
        hits.popleft()


def _ceil(seconds: float) -> int:
    return max(1, int(seconds) + (seconds % 1 > 0))
