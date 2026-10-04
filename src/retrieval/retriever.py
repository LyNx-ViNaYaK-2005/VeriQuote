"""Bounded context selection with exact and near-duplicate suppression."""
import re

from src.models import Hit
from src.retrieval.vector_store import VectorStore


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + 5]) for i in range(max(1, len(words) - 4))}


def retrieve(store: VectorStore, query_vector, top_k: int = 5,
             min_score: float = 0.25, max_chars: int = 7200) -> list[Hit]:
    if not 1 <= top_k <= 10:
        raise ValueError("Top-K must be between 1 and 10.")
    selected = []
    fingerprints = []
    used_chars = 0
    for hit in store.search(query_vector, min(len(store.chunks), top_k * 8)):
        if hit.score < min_score:
            continue
        fingerprint = shingles(hit.chunk.text)
        if any(len(fingerprint & old) / max(1, min(len(fingerprint), len(old))) > 0.8
               for old in fingerprints):
            continue
        if used_chars + len(hit.chunk.text) > max_chars:
            continue
        selected.append(hit)
        fingerprints.append(fingerprint)
        used_chars += len(hit.chunk.text)
        if len(selected) == top_k:
            break
    return selected
