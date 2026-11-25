# ---- Build stage: resolve dependencies with uv ---------------------------------
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first (cached layer), then the project itself
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Pre-download NLTK / tiktoken data so the non-root runtime never writes to site-packages
ENV NLTK_DATA=/app/.cache/nltk TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
RUN /app/.venv/bin/python -c "from llama_index.core.node_parser import SentenceSplitter; \
SentenceSplitter(chunk_size=100, chunk_overlap=10).split_text('Warm up. The caches.')"

# ---- Runtime stage --------------------------------------------------------------
FROM python:3.13-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/.cache /app/.cache
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/app/.cache/nltk \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    UPLOAD_DIR=/app/uploads
RUN mkdir -p /app/uploads && chown app:app /app/uploads
USER app
EXPOSE 8000
CMD ["uvicorn", "rag_production_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
