# Limitations & Design Trade-offs

## Storage and Sessions

- **Session-only storage**: Documents, vectors, FAISS indices, and chat history are held in memory and are lost when a session ends or the service restarts. Export chat to keep a transcript.
- **In-memory FAISS**: Indices are local to a service process. Multiple instances need sticky sessions and do not share vector state.
- **Streamlit hosting**: Short-lived stateless functions are a poor fit for its WebSocket and session requirements.

## PDF Ingestion

- **No OCR**: PyMuPDF extracts digital text; scanned image-only PDFs need OCR before upload.
- **Layout limits**: Complex columns, tables, equations, and diagrams may not extract in visual reading order.
- **Page boundaries**: Chunks never cross pages, so sentences spanning page breaks may be split.

## Embeddings and Retrieval

- **Jina API dependency**: Document indexing and each question's query embedding require outbound access to Jina and a valid `JINA_API_KEY`. Jina availability, quota, and rate limits apply; indexing may fail or be delayed when the service is unavailable.
- **Repeated indexing cost**: Document vectors are reused within the active session, but newly indexed text requires Jina requests. Avoid repeatedly indexing the same documents; exact duplicate PDFs are rejected.
- **Retrieval quality**: Search quality depends on source extraction and chunk boundaries. `MIN_SIMILARITY=0.25` is an uncalibrated cosine cutoff; relevant passages may fall below it.
- **Deduplication trade-off**: Repeated boilerplate can cause valid passages to be suppressed by 5-word shingle overlap.
- **No reranker**: Folio uses one FAISS similarity retrieval stage without a cross-encoder reranker or multi-hop reasoning.

## Generation and Verification

- **Groq API dependency**: Answer generation requires `GROQ_API_KEY`, outbound access, and available Groq quota. Questions and selected passages are sent to Groq.
- **Jina data handling**: Document chunks and questions are sent to Jina for embeddings; consult provider policies for data handling.
- **Verification scope**: Python verifies evidence IDs and exact quote containment. This proves provenance, not that the model's interpretation logically follows from the quote. Inspect citations for important decisions.

## Default Resource Bounds

| Parameter | Limit |
| --- | --- |
| File size | 20 MB per PDF |
| Session uploads | 50 MB total |
| Documents | 8 |
| Pages | 500 per PDF |
| Chunks | 3,000 per session |
| Questions | 40 per session |
| Question length | 1,500 characters |
| Context budget | 7,200 characters |

See [Architecture](architecture.md), [RAG Pipeline](rag-pipeline.md), and [Deployment](deployment.md).
