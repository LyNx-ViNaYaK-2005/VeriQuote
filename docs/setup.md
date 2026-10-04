# Local Setup & Testing

This guide walks through configuring and running Folio locally, running the automated test suite, and executing browser-based QA.

---

## Prerequisites

Before setting up Folio, ensure your system has:

- **Operating System**: Linux, macOS, or Windows
- **Python**: Version `3.11`, `3.12`, or `3.13`
- **Package Manager**: [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended for fast, reproducible dependency installation)
- **Groq API Key**: An active API key from [Groq Console](https://console.groq.com/) for answer generation
- **Hardware**: CPU only (GPU / CUDA is not required)

---

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TheQuantumPanda/folio-rag.git
cd folio-rag
```

### 2. Install Dependencies with `uv`

Run `uv sync` to create a virtual environment and install all dependencies:

```bash
uv sync
```

> [!NOTE]
> Folio specifies CPU-only PyTorch wheels for Linux and Windows in `pyproject.toml` via `[tool.uv.sources]`. `uv sync` automatically installs lightweight CPU binaries without pulling multiple gigabytes of CUDA packages. On macOS, native PyPI wheels with Metal acceleration support are used.

### 3. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Open `.env` in your editor and add your Groq API key:

```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here

# Optional overrides (defaults shown below)
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
MIN_SIMILARITY=0.25
```

#### Environment Variables Reference

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | **Yes** (for chat) | `""` | Server-side API key for Groq LLM inference. |
| `GROQ_MODEL` | No | `openai/gpt-oss-20b` | Groq chat model (must support JSON object output mode). |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face model identifier for SentenceTransformers. |
| `EMBEDDING_DEVICE` | No | `cpu` | Execution device; must remain `cpu`. |
| `MIN_SIMILARITY` | No | `0.25` | Cosine similarity cutoff (range: `-1.0` to `1.0`). |

> [!CAUTION]
> Never commit your `.env` file or expose real API keys in public repositories. `.env` is ignored in [`.gitignore`](../.gitignore).

---

## Running the Application

Start the Streamlit application:

```bash
uv run streamlit run app.py
```

Streamlit will print the local URL (typically `http://localhost:8501`).

### First-Run Model Download Behavior

When you index a document for the first time:
1. SentenceTransformers downloads the model weights for `sentence-transformers/all-MiniLM-L6-v2` (~90 MB).
2. The model files are cached to your system's Hugging Face cache directory (`~/.cache/huggingface/hub` or `$HF_HOME` if set).
3. The model is loaded into memory and cached as a Streamlit resource (`@st.cache_resource`).
4. Subsequent document indexing and queries within the same or new browser sessions reuse the loaded model instantly without re-downloading.

---

## Automated Tests

Folio includes a comprehensive unit and integration test suite in the `tests/` directory.

### Running Unit Tests

Run the full offline test suite:

```bash
uv run --no-sync --offline python -m unittest discover -s tests -v
```

The standard suite uses real generated in-memory PDFs, real FAISS CPU indexing, a stub SentenceTransformer, and a mock Groq client. It runs in under **1 second** without network access or live API calls:

```text
Ran 43 tests in 0.554s

OK (skipped=1)
```

The suite covers:
- PDF extraction (clean, empty, scanned, encrypted, damaged files)
- Page boundaries, chunk generation, and unique chunk IDs
- Normalization, vector shape validation, and cosine mappings
- Shingle repetition suppression and character budgets
- Transactional index building and failed-batch retries
- Citation deduplication and exact-quote containment verification
- Rejection of fabricated quotes, missing IDs, or malformed JSON
- Markdown and plain-text export formatting
- UI session controls and reset dialog workflows

### Optional Real-Model Integration Test

To verify that the real SentenceTransformers model and PyTorch produce unit-normalized `float32` vectors on your CPU, set `RUN_REAL_MODEL_TEST=1`:

```bash
RUN_REAL_MODEL_TEST=1 uv run --no-sync python -m unittest tests/test_embeddings.py -v
```

This test uses `local_files_only=True` and verifies the actual model output shapes and norm constraints.

---

## Browser-Based QA Fixture

For manual end-to-end testing with deterministic mock providers:

```bash
uv run --no-sync --offline streamlit run tests/browser_fixture.py --server.port 8502
```

This starts a mock instance on port 8502 that requires no Groq API key and does not download embedding models. Use it to verify:
1. Uploading PDFs and observing progress status.
2. Asking questions and expanding citation accordions.
3. Exporting chat as Markdown or text.
4. Testing the browser unload alert (`beforeunload`) on reload.
5. Removing individual documents and verifying historical citation preservation.

---

## Acceptance Verification Checklist

Before deploying or presenting Folio:

1. **Index Documents**: Upload two distinct text PDFs; verify page counts and chunk counts match.
2. **Fact Retrieval**: Ask a question with an answer present in Document A; verify the citation points to the correct document and page.
3. **Inspect Evidence**: Expand the citation accordion; verify the supporting quote matches the text and the full passage is viewable.
4. **Fallback Test**: Ask a question about a topic completely absent from the PDFs; verify the fallback message (*"I couldn't find that information in the provided documents."*).
5. **Follow-Up Query**: Ask a follow-up ("Can you elaborate on that?"); verify context from the prior question aids retrieval.
6. **Export Verification**: Click **Export Chat** and download both Markdown and Plain text; ensure citations match the conversation.
7. **Document Removal**: Remove one PDF; verify future questions only retrieve from remaining documents, while older answers retain their citation accordions.

---

## Detailed References

- [System Architecture](architecture.md)
- [RAG Pipeline Deep Dive](rag-pipeline.md)
- [Deployment Guidelines](deployment.md)
- [System Limitations](limitations.md)
