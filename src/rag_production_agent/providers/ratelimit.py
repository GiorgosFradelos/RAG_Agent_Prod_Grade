"""Client-side rate limiting for embedding APIs with tokens/requests-per-minute quotas."""

import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0


def estimate_tokens(text: str) -> int:
    # Conservative (~3 chars/token) so we undershoot the quota rather than overshoot it.
    return max(1, len(text) // 3 + 1)


def make_token_batches(texts: list[str], max_tokens: int) -> list[list[str]]:
    """Group texts, in order, into batches whose estimated token count fits max_tokens.
    A single text larger than max_tokens gets a batch of its own."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        n = estimate_tokens(text)
        if current and current_tokens + n > max_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += n
    if current:
        batches.append(current)
    return batches


def is_rate_limit_error(e: Exception) -> bool:
    msg = f"{type(e).__name__} {e}".lower()
    return any(s in msg for s in ("429", "resourceexhausted", "resource_exhausted", "quota", "rate limit"))


class SlidingWindowLimiter:
    """Thread-safe limiter tracking tokens and requests sent in the last 60 seconds.
    `clock` and `sleep` are injectable so tests don't have to wait."""

    def __init__(
        self,
        tpm: int,
        rpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.tpm = tpm
        self.rpm = rpm
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()

    def acquire(self, tokens: int) -> None:
        tokens = min(tokens, self.tpm)  # an oversized request can still go through on its own
        while True:
            with self._lock:
                now = self._clock()
                self._prune(now)
                used = sum(t for _, t in self._events)
                if used + tokens <= self.tpm and len(self._events) < self.rpm:
                    self._events.append((now, tokens))
                    return
                wait = WINDOW_SECONDS - (now - self._events[0][0]) + 0.05
            logger.info("Embedding rate limit reached, waiting %.1fs", wait)
            self._sleep(wait)

    def penalize(self) -> None:
        """After a 429, treat the current window as full so callers back off."""
        with self._lock:
            self._events.append((self._clock(), self.tpm))


class RateLimitedEmbedder:
    """Wraps a raw `embed_batch` function with batching, rate limiting and retries."""

    def __init__(
        self,
        embed_batch: Callable[[list[str]], list[list[float]]],
        limiter: SlidingWindowLimiter,
        max_retries: int = 6,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._embed_batch = embed_batch
        self._limiter = limiter
        self._max_retries = max_retries
        self._sleep = sleep

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Keep each request to a fraction of the per-minute budget so pacing is smooth.
        request_budget = max(1, self._limiter.tpm // 4)
        vectors: list[list[float]] = []
        for batch in make_token_batches(texts, request_budget):
            vectors.extend(self._embed_with_retry(batch))
        return vectors

    def _embed_with_retry(self, batch: list[str]) -> list[list[float]]:
        tokens = sum(estimate_tokens(t) for t in batch)
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire(tokens)
            try:
                return self._embed_batch(batch)
            except Exception as e:
                if not is_rate_limit_error(e) or attempt == self._max_retries:
                    raise
                self._limiter.penalize()
                backoff = min(60.0, 2**attempt * 2) + random.uniform(0, 1)
                logger.warning("Embedding quota hit (attempt %d), retrying in %.1fs: %s", attempt + 1, backoff, e)
                self._sleep(backoff)
        raise RuntimeError("unreachable")
