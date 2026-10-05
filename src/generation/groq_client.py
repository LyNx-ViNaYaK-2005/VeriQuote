"""Generate structured claims, validate support, then assign citations in Python."""
import json
import logging
import re

from groq import Groq

from src.config import Config
from src.errors import AppError, provider_error
from src.generation.prompts import FALLBACK, SYSTEM_PROMPT
from src.models import Answer, Citation, Hit, Support

logger = logging.getLogger(__name__)


def compact(value: str) -> str:
    return " ".join(value.split())


def validate_answer(payload, hits: list[Hit]) -> Answer:
    """Fail closed: no partially validated generated answer is shown."""
    fallback = Answer(FALLBACK, context=hits)
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        logger.debug("generation verification rejected malformed claims payload; retrieved_passages=%d", len(hits))
        fallback.note = "The model response did not pass evidence validation."
        return fallback
    claims = payload["claims"]
    if not claims:
        logger.debug("generation abstained with no claims; retrieved_passages=%d", len(hits))
        return fallback
    try:
        if len(claims) > 8:
            raise ValueError
        by_id = {f"E{i}": hit for i, hit in enumerate(hits, 1)}
        citations = {}
        sentences = []
        for claim in claims:
            text, evidence = claim["text"], claim["evidence"]
            if (not isinstance(text, str) or not text.strip() or len(text) > 1600
                    or re.search(r"\[\d+\]|https?://", text)
                    or not isinstance(evidence, list) or not 1 <= len(evidence) <= len(hits)):
                raise ValueError
            numbers = []
            for item in evidence:
                hit = by_id[item["id"]]
                quote = item["quote"]
                if not isinstance(quote, str):
                    raise ValueError
                quote = compact(quote)
                passage = compact(hit.chunk.text)
                if len(quote) < min(12, len(passage)) or quote not in passage:
                    raise ValueError
                key = (hit.chunk.document_id, hit.chunk.page)
                if key not in citations:
                    citations[key] = Citation(len(citations) + 1, hit.chunk.document_id,
                                              hit.chunk.source, hit.chunk.page)
                citation = citations[key]
                support = Support(hit, quote)
                if support not in citation.supports:
                    citation.supports.append(support)
                if citation.number not in numbers:
                    numbers.append(citation.number)
            # Escape markup: model output cannot create links, images, or fake UI.
            plain = re.sub(r"([\\`*_{}\[\]<>#!|])", r"\\\1", compact(text))
            sentences.append(plain + " " + " ".join(f"[{n}]" for n in numbers))
        return Answer("\n\n".join(sentences), list(citations.values()), hits)
    except (KeyError, TypeError, ValueError):
        logger.debug("generation verification rejected evidence or quote; retrieved_passages=%d", len(hits))
        fallback.note = "The model response did not pass evidence validation."
        return fallback


class GroqGenerator:
    def __init__(self, config: Config, client=None):
        self.config, self.client = config, client

    def generate(self, question: str, hits: list[Hit], style: str = "Concise",
                 previous_questions: list[str] | None = None) -> Answer:
        if not hits:
            return Answer(FALLBACK)
        if style not in ("Concise", "Detailed", "Explain simply"):
            raise AppError("Choose a supported response style.")
        if self.client is None:
            if not self.config.groq_api_key:
                raise AppError("Add GROQ_API_KEY to the server environment to generate answers.")
            self.client = Groq(api_key=self.config.groq_api_key, timeout=45, max_retries=0)
        request = {
            "question": question, "style": style,
            "previous_questions_for_reference_only": (previous_questions or [])[-2:],
            "evidence": [{"id": f"E{i}", "text": h.chunk.text} for i, h in enumerate(hits, 1)],
        }
        try:
            response = self.client.chat.completions.create(
                model=self.config.groq_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": json.dumps(request, ensure_ascii=False)}],
                response_format={"type": "json_object"}, temperature=0,
                max_completion_tokens=1800,
            )
        except Exception as exc:
            raise provider_error("Groq", exc) from exc
        try:
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                raise ValueError
            payload = json.loads(choice.message.content)
        except (ValueError, TypeError, IndexError, AttributeError):
            logger.debug("generation returned incomplete or invalid JSON; retrieved_passages=%d", len(hits))
            return Answer(FALLBACK, context=hits, note="The model returned an incomplete or invalid response.")
        return validate_answer(payload, hits)
