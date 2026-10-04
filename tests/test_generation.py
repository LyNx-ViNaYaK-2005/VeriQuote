import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.config import Config
from src.generation.groq_client import GroqGenerator, validate_answer
from src.generation.prompts import FALLBACK
from src.models import Chunk, Hit


def hit(identity="a", page=2):
    return Hit(Chunk(identity, "doc", "Retrieval finds relevant passages in a document.", "notes.pdf", page, 0), .8)


def claim(evidence_id="E1", quote="Retrieval finds relevant passages"):
    return {"text": "Retrieval finds passages.", "evidence": [{"id": evidence_id, "quote": quote}]}


class GenerationTests(unittest.TestCase):
    def test_citations_deduplicate_pages_but_keep_support(self):
        answer = validate_answer({"claims": [claim(), claim("E2")]}, [hit(), hit("b")])
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(len(answer.citations[0].supports), 2)
        self.assertEqual(answer.citations[0].source, "notes.pdf")
        self.assertEqual(answer.citations[0].page, 2)

    def test_unknown_citation_fabricated_quote_and_malformed_fail_closed(self):
        for payload in ({"claims": [claim("E99")]}, {"claims": [claim(quote="fabricated quotation")]},
                        {"claims": [claim(), claim(quote="x")]}, {"claims": "wrong"}, None,
                        {"claims": [{"text": "unsupported", "evidence": []}]}):
            with self.subTest(payload=payload):
                answer = validate_answer(payload, [hit()])
                self.assertEqual(answer.text, FALLBACK)
                self.assertEqual(answer.citations, [])

    def test_no_context_never_calls_groq(self):
        client = Mock()
        answer = GroqGenerator(Config(), client).generate("Question", [])
        self.assertEqual(answer.text, FALLBACK)
        client.chat.completions.create.assert_not_called()

    def test_incomplete_and_invalid_json(self):
        for reason, content in (("length", '{"claims": []}'), ("stop", "invalid json")):
            client = Mock()
            client.chat.completions.create.return_value = SimpleNamespace(choices=[
                SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content=content))])
            self.assertEqual(GroqGenerator(Config(), client).generate("q", [hit()]).text, FALLBACK)
