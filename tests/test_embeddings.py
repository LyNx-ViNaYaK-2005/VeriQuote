import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.config import Config
from src.errors import AppError
from src.retrieval.embeddings import SentenceTransformerEmbeddings, load_model, normalized_vectors
from fakes import FakeSentenceTransformer


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        load_model.clear()

    def tearDown(self):
        load_model.clear()

    def test_loader_is_lazy_cached_and_uses_configured_cpu_model(self):
        model = FakeSentenceTransformer()
        constructor = Mock(return_value=model)
        with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=constructor)}):
            first = SentenceTransformerEmbeddings(Config())
            second = SentenceTransformerEmbeddings(Config())
            constructor.assert_not_called()
            first.embed_documents(["document"])
            first.embed_query("question")
            second.embed_documents(["another document"])
            self.assertIs(first.model, second.model)
            constructor.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def test_loader_cache_separates_model_names(self):
        constructor = Mock(side_effect=[FakeSentenceTransformer(), FakeSentenceTransformer()])
        with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=constructor)}):
            first = load_model("first-model", "cpu")
            second = load_model("second-model", "cpu")
            self.assertIs(first, load_model("first-model", "cpu"))
            self.assertIsNot(first, second)
        self.assertEqual(constructor.call_count, 2)

    def test_missing_dependency_is_clear_without_importing_real_package(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaisesRegex(AppError, "uv sync"):
                SentenceTransformerEmbeddings(Config()).embed_documents(["document"])

    def test_model_load_failure_is_safe_and_can_be_retried(self):
        model = FakeSentenceTransformer()
        constructor = Mock(side_effect=[OSError("private-cache-secret"), model])
        with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=constructor)}):
            embedder = SentenceTransformerEmbeddings(Config())
            with self.assertRaises(AppError) as result:
                embedder.embed_query("question")
            self.assertNotIn("private-cache-secret", str(result.exception))
            embedder.embed_query("question")
        self.assertEqual(constructor.call_count, 2)

    def test_batch_reuse_query_dtype_normalization_and_removal(self):
        model = FakeSentenceTransformer()
        embeddings = SentenceTransformerEmbeddings(Config(batch_size=2), model)
        first = embeddings.embed_documents(["a", "b", "c", "a"])
        self.assertEqual([len(call[0]) for call in model.calls], [2, 1])
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(first.flags.c_contiguous)
        self.assertEqual(first.shape, (4, 3))
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(first[0], first[3])
        embeddings.embed_texts(["c", "a"])
        self.assertEqual(len(model.calls), 2)
        query = embeddings.embed_query("question")
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(query.shape, (1, first.shape[1]))
        self.assertEqual(query.dtype, np.float32)
        self.assertTrue(query.flags.c_contiguous)
        np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1, atol=1e-6)
        for _, kwargs in model.calls:
            self.assertEqual(kwargs, {
                "batch_size": 2,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            })
        embeddings.retain(["a"])
        self.assertEqual(len(embeddings.cache), 1)
        embeddings.embed_documents(["a", "b"])
        self.assertEqual(model.calls[-1][0], ["b"])

    def test_default_encode_batch_size_is_32(self):
        model = FakeSentenceTransformer()
        SentenceTransformerEmbeddings(Config(), model).embed_query("question")
        self.assertEqual(model.calls[0][1]["batch_size"], 32)

    def test_empty_text_is_rejected_before_loading(self):
        embeddings = SentenceTransformerEmbeddings(Config())
        with patch("src.retrieval.embeddings.load_model") as loader:
            for texts in ([], [""], ["good", " \n"]):
                with self.subTest(texts=texts), self.assertRaisesRegex(AppError, "empty text"):
                    embeddings.embed_documents(texts)
            with self.assertRaisesRegex(AppError, "empty text"):
                embeddings.embed_query(" ")
            loader.assert_not_called()

    def test_shapes_zero_and_nan_are_rejected(self):
        for vector in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [[[1, 2]]], [[1, 2], [2, 1]]):
            with self.subTest(vector=vector), self.assertRaises(AppError):
                normalized_vectors(vector, 1)

    def test_unexpected_response_shape_is_clean_error(self):
        model = Mock()
        model.encode.return_value = []
        with self.assertRaisesRegex(AppError, "invalid vectors"):
            SentenceTransformerEmbeddings(Config(), model).embed_query("hello")

    def test_encode_error_does_not_leak_text(self):
        model = Mock()
        model.encode.side_effect = RuntimeError("secret-token-and-document")
        with self.assertRaises(AppError) as result:
            SentenceTransformerEmbeddings(Config(), model).embed_query("hello")
        self.assertNotIn("secret-token", str(result.exception))

    def test_retry_reuses_successful_batches(self):
        model = Mock()
        model.encode.side_effect = [[[1, 0]], RuntimeError(), [[0, 1]]]
        embedder = SentenceTransformerEmbeddings(Config(batch_size=1), model)
        with self.assertRaises(AppError):
            embedder.embed_documents(["a", "b"])
        result = embedder.embed_documents(["a", "b"])
        self.assertEqual(model.encode.call_count, 3)
        self.assertEqual([call.args[0] for call in model.encode.call_args_list], [["a"], ["b"], ["b"]])
        np.testing.assert_array_equal(result, [[1, 0], [0, 1]])

    def test_dimensions_are_inferred_and_changes_are_rejected(self):
        model = Mock()
        model.encode.side_effect = [[[1, 0, 0, 0, 0]], [[1, 0]]]
        embedder = SentenceTransformerEmbeddings(Config(), model)
        self.assertEqual(embedder.embed_documents(["a"]).shape, (1, 5))
        with self.assertRaisesRegex(AppError, "changed vector dimensions"):
            embedder.embed_query("question")

    def test_document_text_cache_is_session_owned(self):
        model = FakeSentenceTransformer()
        first = SentenceTransformerEmbeddings(Config(), model)
        second = SentenceTransformerEmbeddings(Config(), model)
        first.embed_documents(["private document"])
        self.assertFalse(second.cache)
        second.embed_documents(["private document"])
        self.assertEqual(len(model.calls), 2)


class RealModelIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("RUN_REAL_MODEL_TEST") == "1", "Opt-in: requires installed dependencies and a cached model")
    def test_cached_real_model_produces_normalized_float32_vectors(self):
        # This optional check is cache-only and never downloads model files.
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                self.skipTest("SentenceTransformers or its dependencies are not installed")
            try:
                model = SentenceTransformer(Config.embedding_model, device="cpu", local_files_only=True)
            except OSError:
                self.skipTest("The embedding model is not available in the local cache")
            embedder = SentenceTransformerEmbeddings(Config(), model)
            documents = embedder.embed_documents(["The sky is blue.", "The grass is green."])
            query = embedder.embed_query("What color is the sky?")
        self.assertEqual(documents.dtype, np.float32)
        self.assertEqual(query.dtype, np.float32)
        self.assertEqual(documents.shape[1], query.shape[1])
        np.testing.assert_allclose(np.linalg.norm(documents, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1, atol=1e-6)
