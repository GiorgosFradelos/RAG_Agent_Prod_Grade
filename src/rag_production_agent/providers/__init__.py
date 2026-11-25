"""Provider factories: pick the embedder / LLM from settings."""

from functools import lru_cache

from rag_production_agent.config import get_settings, secret
from rag_production_agent.providers.base import LLM, Embedder
from rag_production_agent.providers.ratelimit import RateLimitedEmbedder, SlidingWindowLimiter

__all__ = ["LLM", "Embedder", "get_embedder", "get_llm"]


@lru_cache
def get_embedder() -> Embedder:
    """One process-wide embedder, so every caller shares the same rate limiter."""
    s = get_settings()
    if s.embed_provider == "azure":
        from rag_production_agent.providers.azure_openai import azure_embed_batch_fn

        embed_batch = azure_embed_batch_fn(s)
    else:
        from rag_production_agent.providers.gemini import gemini_embed_batch_fn

        embed_batch = gemini_embed_batch_fn(secret(s.gemini_api_key), s.gemini_embed_model)

    limiter = SlidingWindowLimiter(tpm=s.embed_tpm_limit, rpm=s.embed_rpm_limit)
    return RateLimitedEmbedder(embed_batch, limiter, max_retries=s.embed_max_retries)


@lru_cache
def get_llm() -> LLM:
    s = get_settings()
    if s.llm_provider == "azure":
        from rag_production_agent.providers.azure_openai import azure_llm

        return azure_llm(s)
    from rag_production_agent.providers.groq import groq_llm

    return groq_llm(secret(s.groq_api_key), s.groq_model)
