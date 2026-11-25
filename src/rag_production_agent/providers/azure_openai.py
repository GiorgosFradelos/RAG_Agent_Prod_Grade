"""Azure OpenAI (AI Foundry) providers. Needs the optional extra: `uv sync --extra azure`."""

from rag_production_agent.config import Settings, secret
from rag_production_agent.providers.chat import ChatLLM


def _require(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} must be set when using the Azure OpenAI provider")
    return value


def azure_embed_batch_fn(settings: Settings):
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

    deployment = _require(settings.azure_openai_embed_deployment, "AZURE_OPENAI_EMBED_DEPLOYMENT")
    client = AzureOpenAIEmbedding(
        model=settings.azure_openai_embed_model,
        azure_deployment=deployment,
        azure_endpoint=_require(settings.azure_openai_endpoint, "AZURE_OPENAI_ENDPOINT"),
        api_key=secret(settings.azure_openai_api_key),
        api_version=settings.azure_openai_api_version,
    )
    return client.get_text_embedding_batch


def azure_llm(settings: Settings) -> ChatLLM:
    from llama_index.llms.azure_openai import AzureOpenAI

    deployment = _require(settings.azure_openai_chat_deployment, "AZURE_OPENAI_CHAT_DEPLOYMENT")
    llm = AzureOpenAI(
        engine=deployment,
        model=deployment,
        azure_endpoint=_require(settings.azure_openai_endpoint, "AZURE_OPENAI_ENDPOINT"),
        api_key=secret(settings.azure_openai_api_key),
        api_version=settings.azure_openai_api_version,
    )
    return ChatLLM(llm, model=deployment)
