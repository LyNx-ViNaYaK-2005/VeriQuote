import json
import unittest
from unittest.mock import patch

from src.config import Config
from src.errors import AppError
from src.generation.groq_client import GroqGenerator
from src.rag.pipeline import Pipeline
from src.retrieval.embeddings import JinaEmbeddings
from fakes import FakeGroq, FakeJinaClient
from test_ingestion import pdf_bytes


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.model, self.groq = FakeJinaClient(), FakeGroq()
        config = Config(jina_api_key="test")
        self.pipeline = Pipeline(config, JinaEmbeddings(config, self.model), GroqGenerator(config, self.groq))

    def test_full_pdf_to_answer_and_no_document_reembedding(self):
        stages = []
        document = self.pipeline.add_pdf(pdf_bytes("Retrieval finds relevant passages in documents."), "notes.pdf", stages.append)
        self.assertEqual(stages[-1], "Ready")
        document_texts = [chunk.text for chunk in document.chunks]
        self.assertEqual(self.model.calls[0]["input"], document_texts)
        self.pipeline.ask("What does retrieval do?")
        self.pipeline.ask("Explain that simply.")
        self.assertEqual(len(self.model.calls), 3)
        self.assertEqual(sum(call["task"] == "retrieval.passage" for call in self.model.calls), 1)
        self.assertIn("What does retrieval do?", self.model.calls[-1]["input"][0])
        self.assertEqual(self.pipeline.history[0].answer.citations[0].document_id, document.id)
        payload = json.loads(self.groq.calls[-1]["messages"][1]["content"])
        self.assertNotIn("source", payload["evidence"][0])
        self.assertNotIn("vectors", payload)

    def test_bad_file_and_index_failure_preserve_good_documents(self):
        good = self.pipeline.add_pdf(pdf_bytes("First document content."), "first.pdf")
        with self.assertRaises(AppError):
            self.pipeline.add_pdf(b"invalid", "bad.pdf")
        with patch("src.rag.pipeline.VectorStore", side_effect=AppError("index failed")), self.assertRaises(AppError):
            self.pipeline.add_pdf(pdf_bytes("Second document content."), "second.pdf")
        self.assertEqual(list(self.pipeline.documents), [good.id])
        self.assertEqual(len(self.pipeline.store.chunks), len(good.chunks))

    def test_duplicate_name_limits_removal_and_clear(self):
        data = pdf_bytes("First document text.")
        doc = self.pipeline.add_pdf(data, "first.pdf")
        calls_before_duplicate = list(self.model.calls)
        with self.assertRaisesRegex(AppError, "already indexed"):
            self.pipeline.add_pdf(data, "renamed.pdf")
        self.assertEqual(self.model.calls, calls_before_duplicate)
        self.assertEqual(list(self.pipeline.documents), [doc.id])
        with self.assertRaisesRegex(AppError, "filename"):
            self.pipeline.add_pdf(pdf_bytes("Different text."), "first.pdf")
        self.assertEqual(self.model.calls, calls_before_duplicate)
        self.pipeline.ask("What is this?")
        self.pipeline.remove_document(doc.id)
        self.assertIsNone(self.pipeline.store)
        self.assertEqual(self.pipeline.embedder.cache, {})
        self.assertEqual(len(self.pipeline.history), 1)
        self.pipeline.clear_documents()
        with self.assertRaises(AppError):
            self.pipeline.ask("Question")

    def test_adding_another_pdf_does_not_reembed_existing_documents(self):
        first = self.pipeline.add_pdf(pdf_bytes("The first document has useful evidence."), "first.pdf")
        first_vectors = first.vectors.copy()
        second = self.pipeline.add_pdf(pdf_bytes("The second document has different evidence."), "second.pdf")
        self.assertEqual([call["input"] for call in self.model.calls if call["task"] == "retrieval.passage"],
                         [[chunk.text for chunk in first.chunks], [chunk.text for chunk in second.chunks]])
        self.assertTrue((first.vectors == first_vectors).all())
        self.assertEqual(len(self.pipeline.store.chunks), len(first.chunks) + len(second.chunks))

    def test_no_chat_on_generation_failure(self):
        self.pipeline.add_pdf(pdf_bytes("Something worth citing."), "notes.pdf")
        with patch.object(self.pipeline.generator, "generate", side_effect=AppError("failure")), self.assertRaises(AppError):
            self.pipeline.ask("Question")
        self.assertEqual(self.pipeline.history, [])
