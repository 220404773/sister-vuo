from __future__ import annotations

import math
import time
from collections import OrderedDict, deque

from .config import Settings


class SlidingWindowLimiter:
    """Small in-memory sliding window limiter with bounded key storage."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        max_keys: int = 10_000,
    ) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = float(window_seconds)
        self.max_keys = max(1, int(max_keys))
        self._events: OrderedDict[str, deque[float]] = OrderedDict()

    def check(self, key: str, now: float | None = None) -> float | None:
        """Record a request. Return None when allowed, else seconds until retry."""
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        events = self._events.get(key)
        if events is not None:
            while events and events[0] <= cutoff:
                events.popleft()
        if events is not None and len(events) >= self.limit:
            # Keep hot, over-limit keys recent so key-table pressure cannot evict
            # them and silently reset their window.
            self._events.move_to_end(key)
            wait = self.window_seconds - (current - events[0])
            return max(1.0, float(math.ceil(wait)))
        if events is None:
            events = deque()
            self._events[key] = events
        else:
            self._events.move_to_end(key)
        events.append(current)
        while len(self._events) > self.max_keys:
            self._events.popitem(last=False)
        return None


class RateLimiter:
    def __init__(self, settings: Settings) -> None:
        self.start = SlidingWindowLimiter(settings.rate_limit_start_per_minute)
        self.turn = SlidingWindowLimiter(settings.rate_limit_turn_per_minute)
        self.probe = SlidingWindowLimiter(settings.rate_limit_probe_per_minute)
        self.whatsapp = SlidingWindowLimiter(settings.rate_limit_whatsapp_per_minute)
