from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.file import PDFReader




load_dotenv()


# Initialize Gemini's free embedding model (text-embedding-004)
embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # Uses Google's Gemini API for embeddings
    return embed_model.get_text_embedding_batch(texts)