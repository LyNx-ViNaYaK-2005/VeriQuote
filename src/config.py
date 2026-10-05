"""Server configuration; blank optional variables use working defaults."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from src.errors import AppError


@dataclass(frozen=True)
class Config:
    groq_api_key: str = field(default="", repr=False)
    groq_model: str = "openai/gpt-oss-20b"
    jina_api_key: str = field(default="", repr=False)
    embedding_model: str = "jina-embeddings-v3"
    chunk_size: int = 1100
    overlap: int = 200
    batch_size: int = 32
    min_similarity: float = 0.25
    max_file_bytes: int = 20 * 1024 * 1024
    max_session_bytes: int = 50 * 1024 * 1024
    max_documents: int = 8
    max_pages: int = 500
    max_chunks: int = 3000
    context_chars: int = 7200

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise AppError("The embedding batch size must be positive.")

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        def value(name: str, default: str = "") -> str:
            return os.getenv(name, "").strip() or default
        try:
            threshold = float(value("MIN_SIMILARITY", "0.25"))
            if not -1 <= threshold <= 1:
                raise ValueError
        except ValueError as exc:
            raise AppError("MIN_SIMILARITY must be a number between -1 and 1.") from exc
        return cls(
            groq_api_key=value("GROQ_API_KEY"),
            groq_model=value("GROQ_MODEL", cls.groq_model),
            jina_api_key=value("JINA_API_KEY"),
            embedding_model=value("JINA_EMBEDDING_MODEL", cls.embedding_model),
            min_similarity=threshold,
        )
