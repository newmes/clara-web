"""Async RPM rate limiter for API calls."""

import asyncio
import time


class RateLimiter:
    """Token-bucket rate limiter that enforces a minimum interval between calls."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()
