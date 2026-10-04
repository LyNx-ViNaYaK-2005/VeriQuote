"""Shared local CPU model with document vectors cached per browser session."""
import hashlib

import numpy as np
import streamlit as st

from src.config import Config
from src.errors import AppError


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, device: str = "cpu"):
    """Load once per server process; importing this module downloads nothing."""
    if device != "cpu":
        raise AppError("EMBEDDING_DEVICE must be cpu. This deployment uses CPU embeddings.")
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, device=device)
    except ImportError as exc:
        raise AppError("Local embedding dependencies are unavailable. Deploy with uv sync to install them.") from exc
    except Exception as exc:
        raise AppError(
            "The local embedding model could not be loaded. Its first startup needs a model download; "
            "check the server network, model cache, and available memory, then try again."
        ) from exc


def normalized_vectors(values, expected_rows: int) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1 and expected_rows == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
            raise ValueError("Expected one pooled vector per text")
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite embedding")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Invalid norm")
        return np.ascontiguousarray(matrix / norms, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AppError("The embedding model returned invalid vectors. Use a model that returns one sentence vector per input.") from exc


class SentenceTransformerEmbeddings:
    def __init__(self, config: Config, model=None):
        self.config = config
        self.model = model
        self.cache: dict[str, np.ndarray] = {}
        self.dimension: int | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts or any(not text.strip() for text in texts):
            raise AppError("Cannot embed empty text.")
        if self.model is None:
            self.model = load_model(self.config.embedding_model, self.config.embedding_device)
        try:
            values = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            # Model exceptions may include uploaded text or private cache paths.
            raise AppError("The local embedding model could not encode the text. Try fewer or smaller PDFs, then retry.") from exc
        vectors = normalized_vectors(values, len(texts))
        if self.dimension is not None and vectors.shape[1] != self.dimension:
            raise AppError("The embedding model changed vector dimensions. Start a new session.")
        self.dimension = vectors.shape[1]
        return vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts or any(not text.strip() for text in texts):
            raise AppError("Cannot embed empty text.")
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        missing = dict((key, text) for key, text in zip(keys, texts) if key not in self.cache)
        items = list(missing.items())
        for start in range(0, len(items), self.config.batch_size):
            batch = items[start:start + self.config.batch_size]
            vectors = self._encode([text for _, text in batch])
            for (key, _), vector in zip(batch, vectors):
                self.cache[key] = vector
        return np.ascontiguousarray(np.stack([self.cache[key] for key in keys]), dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Compatibility alias for existing callers."""
        return self.embed_documents(texts)

    def retain(self, texts: list[str]) -> None:
        """Discard embeddings for removed documents, retaining shared text."""
        keys = {hashlib.sha256(text.encode()).hexdigest() for text in texts}
        self.cache = {key: vector for key, vector in self.cache.items() if key in keys}

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([query])
