"""
In-process request limits for the single-process server.
"""

import math
import threading
import time
from collections import deque

from starlette.exceptions import HTTPException

WINDOW_S = 60.0


class SlidingWindowLimiter:
    """
    At most `limit` requests in any `WINDOW_S` seconds, across all callers.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit = limit
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """
        Record a request. Returns 0 if it's allowed, otherwise the seconds until it
        would be.
        """
        with self._lock:
            now = time.monotonic()
            while self._hits and now - self._hits[0] >= WINDOW_S:
                self._hits.popleft()
            if len(self._hits) >= self.limit:
                return self._hits[0] + WINDOW_S - now
            self._hits.append(now)
            return 0.0


def retry_after(wait_s: float) -> str:
    return str(max(1, math.ceil(wait_s)))


class BodyLimit:
    """
    Fails the body read with a 413 past `max_bytes` (or the path's entry in
    `route_limits`), including chunked bodies with no Content-Length.
    """

    def __init__(
        self, app, max_bytes: int, route_limits: dict[str, int] | None = None
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.route_limits = route_limits or {}

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.route_limits.get(scope["path"], self.max_bytes)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                # FastAPI re-raises HTTPException from body parsing (others become 400)
                raise HTTPException(413, "body_too_large")
            return message

        await self.app(scope, limited_receive, send)
