"""Small, explicit objects that keep evidence and provenance together."""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Page:
    document_id: str
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    source: str
    page: int
    chunk_index: int


@dataclass
class Document:
    id: str
    source: str
    size: int
    page_count: int
    text_pages: int
    chunks: list[Chunk]
    vectors: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Support:
    hit: Hit
    quote: str


@dataclass
class Citation:
    number: int
    document_id: str
    source: str
    page: int
    supports: list[Support] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    context: list[Hit] = field(default_factory=list)
    note: str = ""


@dataclass
class Turn:
    question: str
    answer: Answer
