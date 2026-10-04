"""Page-bounded character chunks, preferring paragraph and sentence endings."""
import re

from src.models import Chunk, Page


def chunk_pages(pages: list[Page], size: int = 1100, overlap: int = 200) -> list[Chunk]:
    if size < 100 or not 0 <= overlap < size // 2:
        raise ValueError("Chunk size must be >= 100; overlap must be less than half the size.")
    chunks = []
    for page in pages:
        text = page.text.strip()
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                segment = text[start:end]
                boundaries = [m.end() for m in re.finditer(r"\n\n|[.!?](?:[\"')])?\s+", segment)]
                candidates = [b for b in boundaries if b >= size * 0.6]
                if candidates:
                    end = start + candidates[-1]
                else:
                    space = text.rfind(" ", start + size // 2, end)
                    if space != -1:
                        end = space + 1
                # Keep a tiny trailing fragment with this chunk.
                if len(text) - end < 100:
                    end = len(text)
            value = text[start:end].strip()
            if value:
                index = len(chunks)
                chunks.append(Chunk(f"{page.document_id}:p{page.page}:c{index}",
                                    page.document_id, value, page.source, page.page, index))
            if end == len(text):
                break
            next_start = max(start + 1, end - overlap)
            if overlap:
                # Begin overlap at a whole word instead of a severed token.
                space = text.find(" ", next_start, end)
                if space != -1:
                    next_start = space + 1
            start = next_start
    return chunks
