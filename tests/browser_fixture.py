"""Browser QA only: real UI and pipeline, deterministic model and Groq fakes.

Run: uv run streamlit run tests/browser_fixture.py --server.port 8502
Never deploy this entry point. No network inference is performed.
"""
from pathlib import Path
import os
import runpy
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ["GROQ_API_KEY"] = "browser-test-only"

import streamlit as st

from fakes import FakeGroq, FakeSentenceTransformer
from src.config import Config
from src.generation.groq_client import GroqGenerator
from src.rag.pipeline import Pipeline
from src.retrieval.embeddings import SentenceTransformerEmbeddings

config = Config.from_env()
if "pipeline" not in st.session_state:
    st.session_state.pipeline = Pipeline(config, SentenceTransformerEmbeddings(config, FakeSentenceTransformer()), GroqGenerator(config, FakeGroq()))
    st.session_state.upload_version = 0
    st.session_state.exported_turns = 0
    st.session_state.processing_results = []
with patch("src.rag.pipeline.SentenceTransformerEmbeddings", side_effect=lambda cfg: SentenceTransformerEmbeddings(cfg, FakeSentenceTransformer())), \
     patch("src.rag.pipeline.GroqGenerator", side_effect=lambda cfg: GroqGenerator(cfg, FakeGroq())):
    runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
