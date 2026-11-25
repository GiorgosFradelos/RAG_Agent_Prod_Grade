from llama_index.llms.groq import Groq

from rag_production_agent.providers.chat import ChatLLM


def groq_llm(api_key: str | None, model: str) -> ChatLLM:
    return ChatLLM(Groq(model=model, api_key=api_key), model=model)
