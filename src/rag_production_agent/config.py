"""All runtime configuration in one typed place, loaded from the environment / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Providers --------------------------------------------------------------
    embed_provider: Literal["gemini", "azure"] = "gemini"
    llm_provider: Literal["groq", "azure"] = "groq"

    gemini_api_key: SecretStr | None = None
    gemini_embed_model: str = "models/gemini-embedding-001"

    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embed_deployment: str | None = None
    azure_openai_embed_model: str = "text-embedding-3-small"  # the model behind the deployment

    # Must match the embedding model's output size (gemini-embedding-001 → 3072).
    embed_dim: int = 3072

    # --- Embedding rate limits (Gemini free tier defaults) ----------------------
    embed_tpm_limit: int = 25_000
    embed_rpm_limit: int = 80
    embed_max_retries: int = 6

    # --- Chunking & retrieval ---------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    # --- Vector store -----------------------------------------------------------
    # Use ":memory:" for an in-process Qdrant (tests, quick experiments).
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "docs"

    # --- App / orchestration ----------------------------------------------------
    inngest_app_id: str = "rag_app"
    inngest_dev: bool = True  # False in production → Inngest Cloud + signed requests
    upload_dir: Path = Path("uploads")
    api_url: str = "http://localhost:8000"  # used by the Streamlit UI


@lru_cache
def get_settings() -> Settings:
    return Settings()


def secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value else None
