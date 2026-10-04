"""FAISS position i always maps to chunks[i]; inner product of unit vectors is cosine."""
import faiss
import numpy as np

from src.errors import AppError
from src.models import Chunk, Hit
from src.retrieval.embeddings import normalized_vectors


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if not chunks or len({c.id for c in chunks}) != len(chunks):
            raise AppError("The search index requires unique, non-empty chunks.")
        matrix = normalized_vectors(vectors, len(chunks))
        self.chunks = list(chunks)
        try:
            self.index = faiss.IndexFlatIP(matrix.shape[1])
            self.index.add(matrix)
        except Exception as exc:
            raise AppError("The search index could not be built. Try fewer or smaller PDFs.") from exc

    def search(self, query: np.ndarray, limit: int) -> list[Hit]:
        if limit < 1:
            return []
        vector = normalized_vectors(query, 1)
        if vector.shape[1] != self.index.d:
            raise AppError("Query and document embedding dimensions differ. Start a new session after changing models.")
        try:
            scores, positions = self.index.search(vector, min(limit, len(self.chunks)))
            return [Hit(self.chunks[int(i)], float(score))
                    for i, score in zip(positions[0], scores[0]) if i >= 0]
        except Exception as exc:
            raise AppError("The search index could not be queried. Re-index your documents.") from exc
