import json
import unittest
from unittest.mock import patch

import numpy as np

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

    def test_broad_questions_return_grounded_answers_with_page_citations(self):
        document = self.pipeline.add_pdf(
            pdf_bytes("Folio organizes reports into searchable passages.",
                      "Its evidence view connects answers to original pages.",
                      "Readers can export their conversation."), "overview.pdf")
        for question in ("What is this document about?", "Summarize the key ideas."):
            with self.subTest(question=question):
                turn = self.pipeline.ask(question)
                self.assertNotEqual(turn.answer.text, "I couldn't find that information in the provided documents.")
                self.assertTrue(turn.answer.citations)
                self.assertEqual(turn.answer.citations[0].source, document.source)
                self.assertTrue(all(citation.page in (1, 2, 3) for citation in turn.answer.citations))
                self.assertGreaterEqual(len(turn.answer.context), 3)

    def test_specific_supported_and_unrelated_questions(self):
        self.pipeline.add_pdf(pdf_bytes("The evidence inspector links each claim to a quoted passage."), "notes.pdf")
        supported = self.pipeline.ask("What does the evidence inspector link to each claim?")
        self.assertTrue(supported.answer.citations)
        self.assertIn("evidence inspector", supported.answer.text)
        unrelated = self.pipeline.ask("What will the weather be tomorrow?")
        self.assertEqual(unrelated.answer.text, "I couldn't find that information in the provided documents.")

    def test_low_absolute_top_match_reaches_grounded_generation(self):
        class LowScoreEmbedder:
            def embed_documents(self, texts):
                return np.array([[0.20, np.sqrt(0.96)] for _ in texts], dtype=np.float32)

            def embed_query(self, query):
                return np.array([[1.0, 0.0]], dtype=np.float32)

            def retain(self, texts):
                pass

        config = Config(min_similarity=0.25)
        pipeline = Pipeline(config, LowScoreEmbedder(), GroqGenerator(config, FakeGroq()))
        pipeline.add_pdf(pdf_bytes("The report explains evidence-based retrieval."), "report.pdf")
        answer = pipeline.ask("What does the report explain?").answer
        self.assertAlmostEqual(answer.context[0].score, 0.20, places=5)
        self.assertTrue(answer.citations)
        self.assertNotEqual(answer.text, "I couldn't find that information in the provided documents.")

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
