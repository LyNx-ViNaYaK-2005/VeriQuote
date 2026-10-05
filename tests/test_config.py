import os
import unittest
from unittest.mock import patch

from src.config import Config
from src.errors import AppError


class ConfigTests(unittest.TestCase):
    @patch("src.config.load_dotenv")
    def test_defaults_and_provider_keys(self, _load_dotenv):
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()
        self.assertEqual(config.embedding_model, "jina-embeddings-v3")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.groq_model, "openai/gpt-oss-20b")
        self.assertEqual(config.groq_api_key, "")
        self.assertEqual(config.jina_api_key, "")
        self.assertEqual(config.min_similarity, 0.10)

    @patch("src.config.load_dotenv")
    def test_environment_overrides_and_blank_defaults(self, _load_dotenv):
        with patch.dict(os.environ, {
            "GROQ_API_KEY": " test-key ",
            "GROQ_MODEL": " ",
            "JINA_API_KEY": " jina-key ",
            "JINA_EMBEDDING_MODEL": " example/model ",
            "MIN_SIMILARITY": "0.4",
        }, clear=True):
            config = Config.from_env()
        self.assertEqual(config.groq_api_key, "test-key")
        self.assertEqual(config.groq_model, Config.groq_model)
        self.assertEqual(config.embedding_model, "example/model")
        self.assertEqual(config.jina_api_key, "jina-key")
        self.assertEqual(config.min_similarity, 0.4)

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaisesRegex(AppError, "batch size must be positive"):
            Config(batch_size=0)
