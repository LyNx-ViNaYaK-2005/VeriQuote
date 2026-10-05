# Local Setup & Testing

## Prerequisites

- Linux, macOS, or Windows
- Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A [Groq API key](https://console.groq.com/) and [Jina API key](https://jina.ai/embeddings/)
- Outbound internet access for Jina embeddings and Groq generation

No local ML model, PyTorch, or CUDA is required.

## Install and Configure

```bash
git clone https://github.com/TheQuantumPanda/folio-rag.git
cd folio-rag
uv sync
cp .env.example .env
```

Set the keys and defaults in `.env`:

```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
JINA_API_KEY=your_jina_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
JINA_EMBEDDING_MODEL=jina-embeddings-v3
MIN_SIMILARITY=0.25
```

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Yes | empty | Groq answer generation credential. |
| `JINA_API_KEY` | Yes | empty | Jina document and query embedding credential. |
| `GROQ_MODEL` | No | `openai/gpt-oss-20b` | Groq chat model. |
| `JINA_EMBEDDING_MODEL` | No | `jina-embeddings-v3` | Jina embedding model. |
| `MIN_SIMILARITY` | No | `0.25` | Cosine similarity cutoff from -1.0 to 1.0. |

Keep `.env` private and do not commit API keys.

## Run

```bash
uv run streamlit run app.py
```

Open the URL printed by Streamlit, upload text PDFs, and select **Index documents**. Jina is called during indexing; Groq is called when asking a question.

## Automated Tests

Run the offline unit suite with provider fakes:

```bash
uv run --no-sync --offline python -m unittest discover -s tests -v
```

The suite uses generated in-memory PDFs, FAISS, fake Jina/Groq clients, and does not make live API requests.

## Browser QA Fixture

```bash
uv run --no-sync --offline streamlit run tests/browser_fixture.py --server.port 8502
```

The fixture provides deterministic fake provider clients for checking PDF upload, indexing, citations, export, and document removal without API credentials.

## Manual Acceptance Checks

1. Index distinct text PDFs and check page and passage counts.
2. Ask about a fact in a PDF and inspect the cited page and quote.
3. Ask about information absent from the PDFs and check the fallback.
4. Try a follow-up, export the conversation, and remove a document.

See [Architecture](architecture.md) and [Deployment](deployment.md) for more detail.
