# Deployment Guidelines

Folio uses a lightweight service architecture: PyMuPDF parses PDFs, page-bounded chunking prepares text, Jina and Groq provide remote APIs, and FAISS performs in-memory vector search in the service. Deployment no longer needs a local embedding model or large ML runtime.

## Runtime Requirements

- Python 3.11–3.13 and the dependencies in `pyproject.toml`.
- A long-running host that supports Streamlit's WebSocket connections.
- Outbound internet access to Jina's embeddings API and Groq's API.
- `GROQ_API_KEY` and `JINA_API_KEY` configured as deployment secrets/environment variables.
- Enough memory for active PDF text, vectors, FAISS indices, and Streamlit sessions. Usage grows with concurrent sessions and indexed content.

No PyTorch, local embedding model, accelerator, or model download at startup is needed.

## Container and Web Service Hosts

Render, Railway, Fly.io, and similar long-running web service platforms can host Folio. Install with `uv sync` and launch Streamlit on the platform-provided port, for example:

```bash
uv run streamlit run app.py --server.address 0.0.0.0 --server.port "$PORT"
```

Configure both API keys through the platform's secret manager. Configure `GROQ_MODEL`, `JINA_EMBEDDING_MODEL`, and `MIN_SIMILARITY` only when changing defaults. Multiple service instances need sticky sessions: the FAISS index and session data are in-memory and not shared between instances.

## Streamlit Community Cloud

Select a supported Python version and add secrets in **App settings → Secrets**:

```toml
GROQ_API_KEY = "gsk_..."
JINA_API_KEY = "..."
```

Streamlit exposes these secrets to the app environment.

## Serverless Platforms

Traditional short-lived function platforms are a poor fit because Streamlit requires persistent WebSocket connections and Folio keeps FAISS and session state in memory. Provider API limits also apply regardless of hosting platform.

## Operations and Privacy

- Jina calls occur when indexing document chunks and once per question for query embeddings; Groq calls occur for answer generation.
- Provider availability, outbound network access, quotas, and rate limits affect indexing and answers.
- FAISS indices are local to the service process. Sessions and documents are ephemeral and are lost when the session expires or process restarts.
- Uploads and embeddings are not persisted by Folio, but document text sent to Jina and selected passages/questions sent to Groq are subject to each provider's policies.
- Set provider spending controls and monitor service memory for expected concurrent use.

See [Setup](setup.md) for local configuration and [Limitations](limitations.md) for retrieval constraints.
