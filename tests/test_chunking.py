import unittest

from src.ingestion.chunker import chunk_pages
from src.models import Page


class ChunkTests(unittest.TestCase):
    def test_preserves_pages_and_unique_ids(self):
        pages = [Page("doc", "notes.pdf", 1, "Sentence one. " * 150),
                 Page("doc", "notes.pdf", 2, ""), Page("doc", "notes.pdf", 3, "Final page.")]
        chunks = chunk_pages(pages)
        self.assertEqual({c.page for c in chunks}, {1, 3})
        self.assertEqual(len({c.id for c in chunks}), len(chunks))
        self.assertTrue(all(c.source == "notes.pdf" and c.document_id == "doc" for c in chunks))
        self.assertTrue(all(len(c.text) < 1200 for c in chunks))

    def test_no_characters_lost_without_overlap(self):
        text = "A paragraph ends here.\n\n" + "Words and sentences. " * 100
        chunks = chunk_pages([Page("d", "n.pdf", 1, text)], size=200, overlap=0)
        self.assertEqual("".join("".join(c.text.split()) for c in chunks), "".join(text.split()))

    def test_long_unbroken_text_and_invalid_settings(self):
        chunks = chunk_pages([Page("d", "n.pdf", 1, "x" * 5000)])
        self.assertGreater(len(chunks), 1)
        with self.assertRaises(ValueError):
            chunk_pages([], size=100, overlap=90)
