"""Validate and extract text in memory. No OCR and no files written to disk."""
import hashlib
import re
import unicodedata

import pymupdf

from src.config import Config
from src.errors import AppError
from src.models import Page


def safe_filename(name: str) -> str:
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if not unicodedata.category(c).startswith("C"))
    name = re.sub(r'[<>:"|?*]', "_", name).strip(" .")
    return name[:180] or "document.pdf"


def document_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    # Preserve paragraph breaks, collapse extraction-induced single newlines.
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip())


def extract_pdf(data: bytes, filename: str, config: Config) -> list[Page]:
    source = safe_filename(filename)
    if not source.lower().endswith(".pdf"):
        raise AppError("Only PDF files are supported.")
    if not data:
        raise AppError("This file is empty.")
    if len(data) > config.max_file_bytes:
        raise AppError("This PDF exceeds the 20 MB file limit.")
    if not data.lstrip().startswith(b"%PDF-"):
        raise AppError("This file does not have a valid PDF header.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise AppError("This PDF is password-protected. Upload an unlocked copy.")
            if not 0 < len(pdf) <= config.max_pages:
                raise AppError(f"PDFs must contain 1–{config.max_pages} pages.")
            identity = document_id(data)
            pages = [Page(identity, source, i + 1, normalize_text(page.get_text("text", sort=True)))
                     for i, page in enumerate(pdf)]
    except AppError:
        raise
    except Exception as exc:
        raise AppError("This PDF could not be read. It may be damaged or unsupported.") from exc
    if not any(p.text for p in pages):
        raise AppError("No extractable text was found. Scanned PDFs need OCR before uploading.")
    return pages
