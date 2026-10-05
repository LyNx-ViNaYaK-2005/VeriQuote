<p align="center">
  <img src="assets/folio-header.gif" alt="Folio · Document Workspace" width="900">
</p>

<p align="center">
  <strong>A focused workspace for asking questions of your PDFs—and checking the evidence.</strong>
</p>

Folio is an in-memory, session-based Retrieval-Augmented Generation (RAG) application built with Python and Streamlit. It extracts text from PDFs, sends page-bounded chunks to Jina AI for embeddings, searches locally with FAISS, and generates grounded answers through Groq. Every answer is checked in Python for valid evidence IDs and exact quotes, with page-level citations and inspectable source passages. No database or persistent document storage is used.

---

## Screenshots

| Empty Workspace | Active Conversation & Evidence Inspector |
| :---: | :---: |
| ![Folio empty workspace](assets/empty-workspace.png) | ![Folio conversation and evidence](assets/conversation.png) |

*Screenshots captured directly from the running Streamlit UI.*

---

## Key Features

- **Strict Evidence Grounding**: Validates evidence IDs and verifies supporting quotes exist in retrieved passages.
- **Page-Level Citations**: Citations map to document pages and open to supporting quotes and source context.
- **Jina AI Embeddings**: Jina generates document vectors during indexing and query vectors for each question; `JINA_API_KEY` is required.
- **Local FAISS Search**: Exact inner-product search over normalized vectors, kept in session memory.
- **Heuristic Near-Duplicate Filtering**: Suppresses redundant passages using 5-word shingle overlap and similarity thresholds.
- **Session Privacy & Ephemeral State**: PDFs, vectors, and conversations remain in server memory for the active session.
- **Export Transcripts**: Download questions, answers, and page citations as Markdown or plain text.
- **Browser Exit Protection**: Warns users before accidental tab closure when unsaved turns exist.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| **Frontend & UI** | [Streamlit](https://streamlit.io/) |
| **PDF Extraction** | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| **Text Chunking** | Custom page-bounded chunker |
| **Embeddings** | [Jina AI Embeddings API](https://jina.ai/embeddings/) (`jina-embeddings-v3`) |
| **Vector Store** | [FAISS CPU](https://github.com/facebookresearch/faiss) (`IndexFlatIP`) |
| **Generation** | [Groq](https://groq.com/) (`openai/gpt-oss-20b`) |
| **Package Management** | [uv](https://docs.astral.sh/uv/) |

---

## Architecture Summary

```
PDF → PyMuPDF → page-bounded chunking → Jina AI Embeddings API → FAISS
                                                           ↑        ↓
User question → Jina query embedding → retrieval ───────────┘
                                      ↓
                         Groq grounded generation
                                      ↓
                 Python citation verification
                                      ↓
                Grounded answer + citations
```

PDF parsing, chunking, FAISS indexing/search, citation verification, and session state run locally in the service. Jina embeddings and Groq generation are remote API calls. No local ML model, PyTorch, or CUDA is required.

For the complete module breakdown, see the [Architecture Guide](docs/architecture.md).

---

## Quick Local Setup

### Prerequisites

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Groq API key](https://console.groq.com/) and [Jina API key](https://jina.ai/embeddings/)

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Set both required credentials in `.env`:

```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
JINA_API_KEY=your_jina_api_key_here
```

### 3. Run the App

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`, upload text PDFs, click **Index documents**, and ask questions.

---

## Environment Variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | **Yes** | `""` | Server-side Groq generation key. |
| `JINA_API_KEY` | **Yes** | `""` | Server-side Jina embedding key. |
| `GROQ_MODEL` | No | `openai/gpt-oss-20b` | Groq chat model. |
| `JINA_EMBEDDING_MODEL` | No | `jina-embeddings-v3` | Jina embedding model. |
| `MIN_SIMILARITY` | No | `0.10` | Similarity cutoff for retrieved passages; the best positive match is still assessed by Groq when all candidates fall below it. |

---

## Detailed Documentation

- 📐 [**Architecture Guide**](docs/architecture.md)
- 🔄 [**RAG Pipeline Deep Dive**](docs/rag-pipeline.md)
- 🛠️ [**Setup & Testing Guide**](docs/setup.md)
- 🚀 [**Deployment Guidelines**](docs/deployment.md)
- ⚠️ [**Limitations & Trade-offs**](docs/limitations.md)

---

## Known Limitations

- **Session-Only Memory**: Refreshing the page, ending the session, or restarting the service clears documents and conversation.
- **No OCR**: Scanned image-only PDFs need OCR before upload.
- **Provider Dependency**: Indexing and questions require Jina; answer generation also requires Groq. Provider quotas, rate limits, and availability apply.
- **In-Memory FAISS**: Search indices are local to a service process and are not shared across instances.

For details, see [docs/limitations.md](docs/limitations.md).

---

## Project Status

Folio is maintained as an open-source portfolio project demonstrating evidence-grounded RAG, deterministic citation verification, and privacy-conscious session design.
