# RAG Pipeline

Folio preserves page provenance from PDF extraction through answer citations. Parsing, chunking, FAISS search, and citation checks run in the service; embedding and generation requests use Jina and Groq APIs.

## Flow

```text
PDF → PyMuPDF extraction → page-bounded chunking
    → batched document embedding requests to Jina
    → local FAISS indexing
Question → Jina query embedding → FAISS similarity retrieval
    → Groq grounded generation → Python citation verification
    → Grounded answer + page citations + inspectable passages
```

## Stages

1. **PDF extraction**: PyMuPDF reads uploaded PDF bytes in memory, extracts sorted page text, and retains physical page numbers. Scanned image-only PDFs need OCR beforehand.
2. **Page-bounded chunking**: Text is split within each page, preserving source filename, page, and chunk identity. Chunks never cross page boundaries.
3. **Document embeddings**: When documents are indexed, unique chunk texts are sent in batches to Jina using `JINA_EMBEDDING_MODEL` (default `jina-embeddings-v3`). The resulting vectors are normalized and cached in the active session.
4. **FAISS indexing**: Normalized vectors are added to local `IndexFlatIP`; inner product is cosine similarity. FAISS positions map directly to source chunks.
5. **Query embedding**: Each question (including any follow-up query context) is embedded through Jina for that question.
6. **Similarity retrieval**: FAISS ranks passages. `MIN_SIMILARITY`, near-duplicate filtering, and a context character budget limit the selected evidence.
7. **Grounded generation**: Groq receives the question and selected passages and returns structured claims, evidence IDs, and exact quotes.
8. **Citation verification**: Python checks that evidence IDs came from retrieved passages and quotes occur verbatim after whitespace normalization. Citations use source filenames and physical page numbers from the chunks.

## Reuse and Indexing

Document embeddings are created when documents are indexed. Query embeddings are generated per question. Document vectors are cached and reused within the session when overlapping text is indexed again or the FAISS index is rebuilt after document removal. Identical PDFs are rejected before embedding; users should avoid repeatedly indexing the same documents because each newly indexed chunk can incur Jina API use. Session cache is ephemeral and is cleared when the session ends.

## Evidence and Fallback

Retrieved passages receive request-local IDs (`E1`, `E2`, ...). Python rejects malformed responses, unknown evidence IDs, and quotes absent from their cited passages. If no evidence supports a response, Folio returns its safe fallback. Verification establishes citation provenance and quote containment, not semantic entailment; users should inspect cited passages.

See [Architecture](architecture.md) for module responsibilities and [Limitations](limitations.md) for provider and retrieval constraints.
