"""Lightweight middleware to block runaway polling from stale browser tabs."""

import logging
import time

from django.http import HttpResponse

logger = logging.getLogger("perf")

# Stale run IDs whose browser tabs are spamming 404s.
# Add entries here; remove when the tabs are finally closed.
BLOCKED_RUN_IDS = set()


class RequestTimingMiddleware:
    """Log response time for every request. Shows up in docker logs."""

    SLOW_MS = 500  # highlight requests slower than this

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        t0 = time.perf_counter()
        response = self.get_response(request)
        ms = (time.perf_counter() - t0) * 1000
        path = request.get_full_path()
        tag = "SLOW" if ms >= self.SLOW_MS else "OK"
        logger.info("[%s] %6.0fms %s %s → %s",
                    tag, ms, request.method, path, response.status_code)
        return response


class BlockStalePollingMiddleware:
    """Return empty 204 immediately for known-dead run polling requests."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path
        if path.startswith("/api/day/") or path.startswith("/api/sim/"):
            for run_id in BLOCKED_RUN_IDS:
                if run_id in path:
                    return HttpResponse(status=204)
        return self.get_response(request)
