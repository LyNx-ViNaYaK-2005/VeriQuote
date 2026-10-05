"""Bounded context selection with exact and near-duplicate suppression."""
import logging
import re

from src.models import Hit
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

SUMMARY_INTENT = re.compile(
    r"\b(summar(?:y|ize|ise|izing|ising)|overview|what\s+is\s+(?:this|the)\s+document\s+about|"
    r"key\s+ideas|main\s+ideas|high[- ]level\s+overview)\b", re.I
)


def is_broad_question(query: str) -> bool:
    return bool(SUMMARY_INTENT.search(query))


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + 5]) for i in range(max(1, len(words) - 4))}


def retrieve(store: VectorStore, query_vector, top_k: int = 5,
             min_score: float = 0.10, max_chars: int = 7200,
             query_text: str = "", broad: bool = False) -> list[Hit]:
    if not 1 <= top_k <= 10:
        raise ValueError("Top-K must be between 1 and 10.")
    search_limit = len(store.chunks) if broad else min(len(store.chunks), max(top_k * 8, top_k))
    ranked = store.search(query_vector, search_limit)
    selected = []
    fingerprints = []
    used_chars = 0
    limit = min(10, max(top_k, 8)) if broad else top_k

    def append_if_useful(hit: Hit) -> bool:
        nonlocal used_chars
        fingerprint = shingles(hit.chunk.text)
        if any(len(fingerprint & old) / max(1, min(len(fingerprint), len(old))) > 0.8
               for old in fingerprints):
            return False
        if used_chars + len(hit.chunk.text) > max_chars:
            return False
        selected.append(hit)
        fingerprints.append(fingerprint)
        used_chars += len(hit.chunk.text)
        return True

    if broad:
        # First select the best passage from distinct source pages, then fill
        # remaining slots by score. This keeps a compact, representative sample.
        best_by_page = {}
        for hit in ranked:
            best_by_page.setdefault((hit.chunk.document_id, hit.chunk.page), hit)
        for hit in best_by_page.values():
            if len(selected) >= limit:
                break
            append_if_useful(hit)
        selected_ids = {hit.chunk.id for hit in selected}
        for hit in ranked:
            if len(selected) >= limit:
                break
            if hit.chunk.id not in selected_ids:
                append_if_useful(hit)
    else:
        eligible = [hit for hit in ranked if hit.score >= min_score]
        # Keep a positive best match as evidence for the generator to assess
        # when all scores sit below the legacy/global cutoff. A negative best
        # match is not meaningful semantic evidence.
        if not eligible and ranked and ranked[0].score > 0:
            eligible = ranked[:1]
        for hit in eligible:
            append_if_useful(hit)
            if len(selected) >= limit:
                break

    logger.debug(
        "retrieval query=%r indexed_passages=%d top_k=%d top_scores=%s min_similarity=%.4f broad=%s remaining=%d",
        query_text, len(store.chunks), top_k,
        [(hit.score, hit.chunk.source, hit.chunk.page) for hit in ranked[:top_k]],
        min_score, broad, len(selected),
    )
    return selected
