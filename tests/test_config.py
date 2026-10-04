import os
import unittest
from unittest.mock import patch

from src.config import Config
from src.errors import AppError


class ConfigTests(unittest.TestCase):
    @patch("src.config.load_dotenv")
    def test_defaults_require_no_embedding_key(self, _load_dotenv):
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()
        self.assertEqual(config.embedding_model, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(config.embedding_device, "cpu")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.groq_model, "openai/gpt-oss-20b")
        self.assertEqual(config.groq_api_key, "")
        self.assertEqual(config.min_similarity, 0.25)

    @patch("src.config.load_dotenv")
    def test_environment_overrides_and_blank_defaults(self, _load_dotenv):
        with patch.dict(os.environ, {
            "GROQ_API_KEY": " test-key ",
            "GROQ_MODEL": " ",
            "EMBEDDING_MODEL": " example/local-model ",
            "EMBEDDING_DEVICE": " ",
            "MIN_SIMILARITY": "0.4",
        }, clear=True):
            config = Config.from_env()
        self.assertEqual(config.groq_api_key, "test-key")
        self.assertEqual(config.groq_model, Config.groq_model)
        self.assertEqual(config.embedding_model, "example/local-model")
        self.assertEqual(config.embedding_device, "cpu")
        self.assertEqual(config.min_similarity, 0.4)

    @patch("src.config.load_dotenv")
    def test_non_cpu_environment_is_rejected(self, _load_dotenv):
        with patch.dict(os.environ, {"EMBEDDING_DEVICE": "cuda"}, clear=True):
            with self.assertRaisesRegex(AppError, "EMBEDDING_DEVICE must be cpu"):
                Config.from_env()

    def test_non_cpu_config_and_invalid_batch_size_are_rejected(self):
        with self.assertRaisesRegex(AppError, "EMBEDDING_DEVICE must be cpu"):
            Config(embedding_device="cuda")
        with self.assertRaisesRegex(AppError, "batch size must be positive"):
            Config(batch_size=0)
