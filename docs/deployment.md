# Deployment Guidelines

This document outlines deployment considerations for Folio on production and cloud hosting environments. Folio is architected as a lightweight, CPU-first Streamlit application with minimal operational requirements.

---

## Deployment Architecture

Folio requires:
- **Long-Running Process**: A persistent Python process hosting Streamlit.
- **WebSocket Support**: Streamlit communicates with client browsers over WebSockets (`/_stcore/stream`).
- **In-Memory Session State**: Session state lives in the server process memory.
- **CPU Only**: No GPU or CUDA drivers are required.

```
[Browser Client]
       │
       ▼ (HTTPS / WSS)
[Reverse Proxy / Cloud Host]
       │
       ▼
[Streamlit Server (app.py)]
   ├─ Session State (Pipeline, FAISS, History)
   ├─ Cached Model (SentenceTransformers CPU)
   └─ Outbound HTTPS ──► [Groq API]
```

---

## Build and Start Commands

For Linux container environments, use the following build and start commands:

### Build Command

```bash
uv sync --locked --no-dev
```

- `--locked`: Asserts that `uv.lock` is strictly respected, ensuring deterministic dependencies across environments.
- `--no-dev`: Skips development tools or test fixtures if defined.

### Start Command

```bash
uv run --no-sync streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}" --server.headless true
```

- `--no-sync`: Prevents `uv` from checking package indexes or mutating the environment at container runtime.
- `--server.headless true`: Configures Streamlit to run in server mode without attempting to open local browser windows.
- `--server.address 0.0.0.0`: Binds the server to all interfaces.

---

## Hardware & Resource Sizing

### CPU and Memory Requirements

| Resource | Minimum | Recommended | Notes |
| --- | --- | --- | --- |
| **CPU** | 1 vCPU | 2 vCPU | Ingestion and cosine search are fast on modern CPUs; embedding generation scales with CPU cores. |
| **RAM** | 1 GB | 2 GB | Baseline process with PyTorch CPU and Streamlit consumes ~450–600 MB. Document vectors and active sessions need additional buffer. |
| **Disk** | 2 GB | 5 GB | Accommodates Python packages (~800 MB with PyTorch CPU) and the Hugging Face model cache (~90 MB). |
| **GPU** | None | None | `EMBEDDING_DEVICE=cpu` is strictly enforced. |

### Cold Starts & Model Warmup

- **First Ingestion Latency**: The first time any user selects **Index documents**, SentenceTransformers downloads and loads `sentence-transformers/all-MiniLM-L6-v2` into memory. This initial load can take 5–15 seconds depending on network throughput.
- **Warm Reruns**: Once loaded, `@st.cache_resource` preserves the model in memory across Streamlit reruns and multiple user sessions.
- **Persistent Cache (`HF_HOME`)**: If your hosting platform supports persistent disks, set `HF_HOME=/path/to/persistent/cache` so the model files persist across container restarts.

---

## PyTorch & Binary Sizing

Standard PyTorch installations with CUDA support exceed 2–4 GB. Folio avoids this bloat:

- `pyproject.toml` pins PyTorch to the official CPU wheel index on Linux and Windows:
  ```toml
  [tool.uv.sources]
  torch = [
      { index = "pytorch-cpu", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
  ]

  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true
  ```
- This keeps the entire virtual environment under ~800 MB on disk.

---

## Hosting Platform Considerations

### Container & Web Services (e.g. Render, Railway, Fly.io)

Container hosts that support long-running processes and WebSockets are well-suited for Folio:
- Ensure the port is mapped using the `$PORT` environment variable.
- Configure outbound network access to:
  - `https://huggingface.co` (for downloading the embedding model on first start, unless pre-baked into the image).
  - `https://api.groq.com` (for LLM inference).
- Scale with a single instance or sticky sessions. Because session state is kept in memory, round-robin load balancing across multiple instances will lose conversational state between requests.

### Streamlit Community Cloud

Streamlit Community Cloud natively supports Streamlit applications:
- Select Python 3.11 in the repository settings.
- Add your `GROQ_API_KEY` under **App settings → Secrets** as a root TOML key:
  ```toml
  GROQ_API_KEY = "gsk_..."
  ```
  Streamlit automatically injects root secrets into `os.environ`.

### Serverless Functions (e.g. Vercel, AWS Lambda)

> [!WARNING]
> Folio is **not** designed for traditional serverless architectures like Vercel Functions or AWS Lambda:
> - Streamlit requires a persistent, bi-directional WebSocket connection.
> - Serverless execution environments spin down quickly, which destroys in-memory FAISS indices and session history.
> - Cold start penalties and bundle size limitations in serverless runtimes conflict with PyTorch and local model execution.

---

## Environment Variables Configuration

Set these variables in your hosting environment:

```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
MIN_SIMILARITY=0.25
```

### Security & Operational Best Practices

1. **Secret Isolation**: Store `GROQ_API_KEY` securely as a backend environment variable. It is never exposed in the client UI, exported files, or error messages.
2. **Operator Budget Protection**: Anonymous visitors interact with the app using the operator's Groq key. While Folio enforces per-session guards (40 questions, 1,500 characters, 50 MB total upload), configure spending caps directly in your [Groq Console](https://console.groq.com/).
3. **Session Lifecycle**: Uploaded documents and vectors stay in memory only for the duration of the Streamlit session. When a user closes the browser or the session expires, Python garbage collection frees the associated memory.

---

## Detailed References

- [System Architecture](architecture.md)
- [RAG Pipeline Deep Dive](rag-pipeline.md)
- [Local Setup & Testing Guide](setup.md)
- [System Limitations](limitations.md)
