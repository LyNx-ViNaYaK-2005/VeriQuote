from datetime import datetime, timezone
import unittest

from src.export.chat_export import export_chat
from src.models import Answer, Citation, Turn


class ExportTests(unittest.TestCase):
    def test_markdown_contains_conversation_and_only_public_fields(self):
        history = [Turn("Question?", Answer("An answer. [1]", [Citation(1, "internal-secret-id", "notes.pdf", 2)]))]
        result = export_chat(history, ["other.pdf"], timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
        for value in ("2026-01-01", "Question?", "An answer. [1]", "notes.pdf — Page 2", "other.pdf"):
            self.assertIn(value, result)
        self.assertNotIn("internal-secret-id", result)
        self.assertNotIn("api_key", result)

    def test_plain_text_and_empty(self):
        self.assertIn("None", export_chat([], []))
        self.assertFalse(export_chat([], [], format="txt").startswith("#"))
