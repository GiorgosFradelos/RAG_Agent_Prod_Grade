import logging
import os
import random
import threading
import time
from collections import deque

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.file import PDFReader


load_dotenv()

logger = logging.getLogger("uvicorn")

# Initialize Gemini's embedding model
embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# --- Rate limiting -----------------------------------------------------------
# Free-tier quotas for the Gemini embedding API. Override in .env if your
# project's limits differ. We stay a bit below the real quota on purpose.
EMBED_TPM_LIMIT = int(os.getenv("EMBED_TPM_LIMIT", "25000"))   # tokens / minute
EMBED_RPM_LIMIT = int(os.getenv("EMBED_RPM_LIMIT", "80"))      # requests / minute
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "6"))
WINDOW_SECONDS = 60.0


def estimate_tokens(text: str) -> int:
    # Conservative estimate (~3 chars/token) so we undershoot the quota rather than overshoot it.
    return max(1, len(text) // 3 + 1)


class _SlidingWindowLimiter:
    """Process-wide limiter tracking tokens and requests sent in the last 60s."""

    def __init__(self, tpm: int, rpm: int):
        self.tpm = tpm
        self.rpm = rpm
        self.events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self.lock = threading.Lock()

    def _prune(self, now: float):
        while self.events and now - self.events[0][0] >= WINDOW_SECONDS:
            self.events.popleft()

    def acquire(self, tokens: int):
        tokens = min(tokens, self.tpm)  # a single oversized request can still go through alone
        while True:
            with self.lock:
                now = time.monotonic()
                self._prune(now)
                used = sum(t for _, t in self.events)
                if used + tokens <= self.tpm and len(self.events) < self.rpm:
                    self.events.append((now, tokens))
                    return
                # Wait until the oldest entry leaves the window
                wait = WINDOW_SECONDS - (now - self.events[0][0]) + 0.05
            logger.info(f"Embedding rate limit reached, waiting {wait:.1f}s")
            time.sleep(wait)

    def penalize(self):
        """After a 429, treat the current window as full so we back off."""
        with self.lock:
            self.events.append((time.monotonic(), self.tpm))


_limiter = _SlidingWindowLimiter(EMBED_TPM_LIMIT, EMBED_RPM_LIMIT)


def _is_rate_limit_error(e: Exception) -> bool:
    msg = f"{type(e).__name__} {e}".lower()
    return any(s in msg for s in ("429", "resourceexhausted", "resource_exhausted", "quota", "rate limit"))


def make_token_batches(texts: list[str], max_tokens: int | None = None) -> list[list[str]]:
    """Group texts into batches whose estimated token count fits within max_tokens."""
    max_tokens = max_tokens or EMBED_TPM_LIMIT
    batches, current, current_tokens = [], [], 0
    for t in texts:
        n = estimate_tokens(t)
        if current and current_tokens + n > max_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(t)
        current_tokens += n
    if current:
        batches.append(current)
    return batches


def _embed_with_retry(batch: list[str]) -> list[list[float]]:
    tokens = sum(estimate_tokens(t) for t in batch)
    for attempt in range(EMBED_MAX_RETRIES + 1):
        _limiter.acquire(tokens)
        try:
            return embed_model.get_text_embedding_batch(batch)
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt == EMBED_MAX_RETRIES:
                raise
            _limiter.penalize()
            backoff = min(60.0, 2 ** attempt * 2) + random.uniform(0, 1)
            logger.warning(f"Embedding quota hit (attempt {attempt + 1}), retrying in {backoff:.1f}s: {e}")
            time.sleep(backoff)
    raise RuntimeError("unreachable")


# --- Public API ----------------------------------------------------------------

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with Gemini, respecting the tokens/requests-per-minute quota."""
    # Each request is kept to a fraction of the TPM budget so the limiter can pace smoothly.
    vectors: list[list[float]] = []
    for batch in make_token_batches(texts, max_tokens=max(1, EMBED_TPM_LIMIT // 4)):
        vectors.extend(_embed_with_retry(batch))
    return vectors
