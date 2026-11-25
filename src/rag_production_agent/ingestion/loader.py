"""PDF loading and sentence-aware chunking."""

from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader


def load_pdf_text(path: str | Path) -> list[str]:
    """Return the text of each page, skipping empty pages."""
    docs = PDFReader().load_data(file=Path(path))
    return [d.text for d in docs if getattr(d, "text", None)]


def chunk_texts(texts: list[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[str] = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks


def load_and_chunk_pdf(path: str | Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    return chunk_texts(load_pdf_text(path), chunk_size, chunk_overlap)
