"""
In-process request limits for the single-process server.
"""

import math
import threading
import time
from collections import deque
from collections.abc import Callable

from starlette.exceptions import HTTPException


class SlidingWindowLimiter:
    """
    At most `limit` requests in any `window_s` seconds, across all callers.
    """

    def __init__(
        self,
        limit: int,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """
        Record a request. Returns 0 if it's allowed, otherwise the seconds until it
        would be.
        """
        with self._lock:
            now = self._clock()
            while self._hits and now - self._hits[0] >= self.window_s:
                self._hits.popleft()
            if len(self._hits) >= self.limit:
                return self._hits[0] + self.window_s - now
            self._hits.append(now)
            return 0.0


def retry_after(wait_s: float) -> str:
    return str(max(1, math.ceil(wait_s)))


class BodyLimit:
    """
    Fails the body read with a 413 past `max_bytes`, including chunked bodies with no
    Content-Length.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                # FastAPI re-raises HTTPException from body parsing (others become 400)
                raise HTTPException(413, "body_too_large")
            return message

        await self.app(scope, limited_receive, send)


class Slots:
    """
    Non-blocking cap on concurrent inferences.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        self._sem = threading.BoundedSemaphore(n)

    def try_acquire(self) -> bool:
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        self._sem.release()
