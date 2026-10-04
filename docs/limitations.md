# Limitations & Design Trade-offs

Folio is designed as a focused, lightweight, and inspectable portfolio RAG application. To prioritize architectural simplicity, data privacy, and zero-database deployment, deliberate design trade-offs were made.

---

## Architectural & Storage Limitations

### 1. Ephemeral, Session-Only Storage
- **No Database**: Folio does not persist documents, embeddings, or chat histories to PostgreSQL, SQLite, or disk.
- **Session Evaporation**: All state is held in `st.session_state`. Refreshing the browser, closing the tab, restarting the server process, or experiencing a network disconnection wipes the active workspace.
- **Export Necessity**: Users must explicitly click **Export Chat** to download Markdown or plain text transcripts before ending a session.

### 2. Multi-Instance & Serverless Limitations
- **No Distributed State**: The FAISS index and document vectors are held in the local Python process heap. Running multiple backend instances behind a round-robin load balancer without sticky sessions will result in lost session state between user interactions.
- **Incompatible with Serverless Functions**: Folio cannot run on stateless function platforms (like AWS Lambda or Vercel Functions) due to Streamlit's WebSocket requirement, PyTorch bundle sizes, and the need for persistent in-memory session state.

---

## Document Parsing & Ingestion Limitations

### 1. No OCR (Optical Character Recognition)
- Folio uses PyMuPDF's digital text extraction (`page.get_text("text", sort=True)`).
- **Scanned or Image-Only PDFs**: PDFs consisting solely of scanned images or rasterized pages will be rejected during indexing with an explicit error:
  > *"No extractable text was found. Scanned PDFs need OCR before uploading."*
- Documents must be pre-processed with an external OCR tool (e.g., Tesseract or Adobe Acrobat) prior to uploading.

### 2. Layout, Column, and Table Parsing
- While PyMuPDF uses coordinate sorting (`sort=True`) to read multi-column layouts naturally, complex magazine layouts, multi-column scientific papers, nested tables, mathematical equations, and diagrams may still be extracted out of visual reading order.
- Folio does not perform layout analysis (such as table structure recognition or bounding box figure extraction).

### 3. Page-Bounded Chunking Boundaries
- Chunks strictly terminate at page boundaries to ensure that every citation maps deterministically to a physical page number.
- Sentences or paragraphs that bridge across page boundaries are split at the page boundary.

---

## Embedding & Retrieval Limitations

### 1. Local CPU Embedding Overhead
- Running `sentence-transformers/all-MiniLM-L6-v2` locally on CPU eliminates external embedding API costs and quota concerns, but:
  - Adds ~800 MB to the application's runtime footprint (PyTorch CPU + model weights).
  - Can cause an initial cold-start delay (5–15 seconds) when downloading model weights on first indexing.
  - Slower indexing throughput on low-spec single-core CPUs compared to high-throughput cloud embedding APIs.

### 2. Retrieval Quality & Threshold Sensitivities
- **Similarity Threshold (`MIN_SIMILARITY = 0.25`)**: The cutoff score is model-dependent and represents an uncalibrated cosine similarity, not a factual confidence probability. Passages with lower semantic overlap—even if factually pertinent—may be excluded.
- **5-Word Shingle Deduplication**: If multiple documents share significant repetitive phrasing (such as standardized headers or boilerplate legalese), valid passages may be filtered out as near-duplicates (>80% shingle overlap).
- **No Reranker**: Folio uses single-stage bi-encoder retrieval (FAISS cosine search). It does not include a secondary cross-encoder reranker, multi-hop reasoning agent, or query expansion step.

---

## Generation & Verification Guarantees

### 1. External Groq API Dependency
- While document indexing and embedding run 100% locally and offline, **answering questions requires an active Groq API key and outbound internet access**.
- Queries are subject to Groq API availability, network latency, and provider rate limits.
- Sent passages and queries are subject to Groq's data and privacy policies.

### 2. Provenance vs. Semantic Entailment
- Folio's Python-side validation checks **citation provenance and exact quote containment**:
  - It confirms that every cited evidence ID was in the retrieved set.
  - It confirms that every quoted string appears verbatim in the source passage.
- **Important**: This does **not** mathematically prove semantic entailment. An LLM can theoretically cite a real sentence from a document while subtly misinterpreting its contextual nuance or drawing an invalid inference. Users should always click and inspect the cited excerpt in the UI before relying on critical answers.

---

## Summary of Resource Bounds

Default server-side bounds (configured in [`src/config.py`](../src/config.py)):

| Parameter | Limit | Reason |
| --- | --- | --- |
| Max File Size | 20 MB / PDF | Prevents memory exhaustion during PDF parsing. |
| Max Session Size | 50 MB total | Bounds per-session memory consumption. |
| Max Documents | 8 PDFs | Keeps indexing fast and UI uncluttered. |
| Max Pages | 500 pages / PDF | Avoids excessive chunk generation on CPU. |
| Max Chunks | 3,000 per session | Prevents excessive FAISS memory usage. |
| Max Questions | 40 per session | Limits provider API cost per session. |
| Question Length | 1,500 characters | Prevents prompt bloating. |
| Context Budget | 7,200 characters | Keeps prompt token count compact and focused. |

---

## Detailed References

- [System Architecture](architecture.md)
- [RAG Pipeline Deep Dive](rag-pipeline.md)
- [Local Setup & Testing Guide](setup.md)
- [Deployment Guidelines](deployment.md)
