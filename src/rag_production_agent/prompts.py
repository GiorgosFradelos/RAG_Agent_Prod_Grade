"""Prompt templates. Bump PROMPT_VERSION whenever the wording changes so eval runs
and traces can be tied to the exact prompt that produced them."""

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "If the context does not contain the answer, say you don't know."
)

NO_CONTEXT_ANSWER = "I couldn't find anything relevant to that question in the uploaded documents."


def build_user_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )
