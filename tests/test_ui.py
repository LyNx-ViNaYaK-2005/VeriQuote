import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.config import Config
from src.generation.groq_client import GroqGenerator
from src.rag.pipeline import Pipeline
from src.retrieval.embeddings import SentenceTransformerEmbeddings
from fakes import FakeGroq, FakeSentenceTransformer
from test_ingestion import pdf_bytes


class UITests(unittest.TestCase):
    def setUp(self):
        self.config = Config(groq_api_key="test")
        self.env = patch("src.config.Config.from_env", return_value=self.config)
        self.env.start()
        self.addCleanup(self.env.stop)
        # AppTest creates a new runtime per test, but imported component modules
        # are cached. Exercise the JS boundary in a real browser, not AppTest.
        guard = patch("src.ui.browser_events.protect_session")
        guard.start()
        self.addCleanup(guard.stop)

    def test_empty_state(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.chat_input), 0)
        self.assertEqual(app.title[0].value, "Read deeper. Ask better.")

    def test_indexing_without_api_keys_keeps_chat_disabled(self):
        data = pdf_bytes("Local embeddings preserve this passage for retrieval.")
        upload = SimpleNamespace(name="local.pdf", size=len(data), getvalue=lambda: data)
        model = FakeSentenceTransformer()
        # AppTest cannot populate file_uploader; represent the pending upload
        # until indexing advances the uploader's version and clears it.
        def uploaded_files(*args, **kwargs):
            return [upload] if kwargs["key"] == "uploads_0" else []

        with patch("src.config.Config.from_env", return_value=Config()), \
             patch("streamlit.file_uploader", side_effect=uploaded_files), \
             patch("src.retrieval.embeddings.load_model", return_value=model):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            index_button = next(button for button in app.button if button.label == "Index documents")
            self.assertFalse(index_button.disabled)
            index_button.click().run()
            self.assertEqual(len(app.exception), 0)
            pipeline = app.session_state.pipeline
            self.assertEqual([doc.source for doc in pipeline.documents.values()], ["local.pdf"])
            self.assertIsNotNone(pipeline.store)
            self.assertEqual(len(model.calls), 1)
            self.assertTrue(app.session_state.processing_results[0][0])
            self.assertTrue(app.chat_input[0].disabled)

    def test_workspace_question_removal_and_reset_confirmation(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        pipeline = Pipeline(self.config, SentenceTransformerEmbeddings(self.config, FakeSentenceTransformer()), GroqGenerator(self.config, FakeGroq()))
        doc = pipeline.add_pdf(pdf_bytes("Folio retrieves passages and preserves source pages."), "notes.pdf")
        app.session_state.pipeline = pipeline
        app.run()
        self.assertEqual(len(app.exception), 0)
        app.chat_input[0].set_value("What does Folio do?").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(pipeline.history), 1)
        app.button(key=f"remove_{doc.id}").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(pipeline.history), 1)
        self.assertTrue(app.chat_input[0].disabled)
        next(b for b in app.button if b.label == "New session").click().run()
        self.assertEqual(len(pipeline.history), 1)
        self.assertTrue(any(b.label == "Clear and start new" for b in app.button))
        next(b for b in app.button if b.label == "Clear and start new").click().run()
        self.assertEqual(len(app.session_state.pipeline.history), 0)
        self.assertEqual(len(app.exception), 0)
