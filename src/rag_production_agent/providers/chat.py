"""Adapter from LlamaIndex chat LLMs to our `LLM` protocol."""

from llama_index.core.llms import ChatMessage, MessageRole

from rag_production_agent.models import Generation


def _usage(raw) -> tuple[int | None, int | None]:
    usage = getattr(raw, "usage", None) or (raw.get("usage") if isinstance(raw, dict) else None)
    if usage is None:
        return None, None
    get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)
    return get("prompt_tokens"), get("completion_tokens")


class ChatLLM:
    """Adapter from any LlamaIndex chat LLM to our `LLM` protocol."""

    def __init__(self, llm, model: str):
        self._llm = llm
        self.model = model

    def generate(self, system: str, user: str) -> Generation:
        response = self._llm.chat(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=system),
                ChatMessage(role=MessageRole.USER, content=user),
            ]
        )
        input_tokens, output_tokens = _usage(response.raw)
        return Generation(
            text=(response.message.content or "").strip(),
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
