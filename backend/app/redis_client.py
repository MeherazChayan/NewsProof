from __future__ import annotations

import time
from collections import defaultdict, deque

from redis import Redis

from app.config import get_settings


class RateLimiter:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._fallback = defaultdict(deque)
        try:
            self.redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
            self.redis.ping()
        except Exception:
            self.redis = None

    def allow(self, key: str, limit: int | None = None, window_seconds: int | None = None) -> bool:
        if not self.settings.rate_limit_enabled:
            return True
        if self.settings.app_env == "development":
            return True
        if key in {"127.0.0.1", "::1", "localhost", "testclient"}:
            return True

        limit = limit or self.settings.rate_limit_requests
        window_seconds = window_seconds or self.settings.rate_limit_window_seconds
        if self.redis:
            redis_key = f"rate-limit:{key}"
            current = self.redis.incr(redis_key)
            if current == 1:
                self.redis.expire(redis_key, window_seconds)
            return current <= limit

        now = time.time()
        entries = self._fallback[key]
        while entries and now - entries[0] > window_seconds:
            entries.popleft()
        if len(entries) >= limit:
            return False
        entries.append(now)
        return True


rate_limiter = RateLimiter()
