"""Session-owned orchestration. Documents commit only after successful indexing."""
import re
from collections.abc import Callable

import numpy as np

from src.config import Config
from src.errors import AppError
from src.generation.groq_client import GroqGenerator
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import document_id, extract_pdf, safe_filename
from src.models import Document, Turn
from src.retrieval.embeddings import SentenceTransformerEmbeddings
from src.retrieval.retriever import retrieve
from src.retrieval.vector_store import VectorStore


class Pipeline:
    def __init__(self, config: Config, embedder=None, generator=None):
        self.config = config
        self.embedder = embedder or SentenceTransformerEmbeddings(config)
        self.generator = generator or GroqGenerator(config)
        self.documents: dict[str, Document] = {}
        self.store: VectorStore | None = None
        self.history: list[Turn] = []

    def _build_store(self, documents: list[Document]) -> VectorStore | None:
        if not documents:
            return None
        return VectorStore([c for d in documents for c in d.chunks],
                           np.concatenate([d.vectors for d in documents]))

    def add_pdf(self, data: bytes, filename: str,
                progress: Callable[[str], None] = lambda stage: None) -> Document:
        identity = document_id(data)
        if identity in self.documents:
            raise AppError("This PDF is already indexed; no new embeddings were generated.")
        if len(self.documents) >= self.config.max_documents:
            raise AppError(f"A session supports up to {self.config.max_documents} PDFs.")
        if sum(d.size for d in self.documents.values()) + len(data) > self.config.max_session_bytes:
            raise AppError("The session upload limit is 50 MB. Remove a document first.")
        source = safe_filename(filename)
        if any(d.source == source for d in self.documents.values()):
            raise AppError("A different PDF has this filename. Rename it to keep citations unambiguous.")
        progress("Extracting text")
        pages = extract_pdf(data, source, self.config)
        progress("Creating page-aware chunks")
        chunks = chunk_pages(pages, self.config.chunk_size, self.config.overlap)
        if len(chunks) + sum(len(d.chunks) for d in self.documents.values()) > self.config.max_chunks:
            raise AppError(f"This exceeds the {self.config.max_chunks:,}-chunk session limit. Use shorter PDFs.")
        progress(f"Generating embeddings · {len(chunks):,} passages")
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        document = Document(identity, source, len(data), len(pages), sum(bool(p.text) for p in pages),
                            chunks, vectors)
        progress("Building search index")
        store = self._build_store([*self.documents.values(), document])
        self.documents[identity] = document
        self.store = store
        progress("Ready")
        return document

    def remove_document(self, identity: str) -> None:
        remaining = {k: v for k, v in self.documents.items() if k != identity}
        store = self._build_store(list(remaining.values()))
        self.documents, self.store = remaining, store
        self.embedder.retain([c.text for d in remaining.values() for c in d.chunks])

    def clear_documents(self) -> None:
        self.documents.clear()
        self.store = None
        self.embedder.retain([])

    def ask(self, question: str, top_k: int = 5, style: str = "Concise",
            follow_up: bool = True) -> Turn:
        question = question.strip()
        if not question:
            raise AppError("Enter a question about your documents.")
        if len(question) > 1500:
            raise AppError("Keep your question under 1,500 characters.")
        if self.store is None:
            raise AppError("Index at least one PDF before asking a question.")
        if len(self.history) >= 40:
            raise AppError("This session has reached 40 questions. Export your chat and start a new session.")
        previous = [turn.question for turn in self.history[-2:]] if follow_up else []
        query = question
        if previous and re.search(r"\b(it|its|they|them|their|this|that|these|those|more|elaborate)\b", question, re.I):
            query = f"{previous[-1][:500]}\nFollow-up: {question}"
        hits = retrieve(self.store, self.embedder.embed_query(query), top_k,
                        self.config.min_similarity, self.config.context_chars)
        answer = self.generator.generate(question, hits, style, previous)
        turn = Turn(question, answer)
        self.history.append(turn)
        return turn
