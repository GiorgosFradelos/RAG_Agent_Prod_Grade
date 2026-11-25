from llama_index.embeddings.gemini import GeminiEmbedding


def gemini_embed_batch_fn(api_key: str | None, model: str):
    client = GeminiEmbedding(model_name=model, api_key=api_key)
    return client.get_text_embedding_batch
