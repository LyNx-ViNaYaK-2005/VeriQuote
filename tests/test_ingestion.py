import unittest

import pymupdf

from src.config import Config
from src.errors import AppError
from src.ingestion.pdf_loader import extract_pdf, safe_filename


def pdf_bytes(*texts):
    with pymupdf.open() as pdf:
        for text in texts:
            page = pdf.new_page()
            if text:
                page.insert_text((72, 72), text)
        return pdf.tobytes()


class PDFTests(unittest.TestCase):
    def test_real_pdf_pages_and_blanks(self):
        pages = extract_pdf(pdf_bytes("First page", "", "Third page"), "notes.pdf", Config())
        self.assertEqual([p.page for p in pages], [1, 2, 3])
        self.assertEqual([p.text for p in pages], ["First page", "", "Third page"])
        self.assertEqual(len({p.document_id for p in pages}), 1)

    def test_bad_empty_scanned_and_oversize(self):
        for data in (b"", b"hello", b"%PDF-broken", pdf_bytes("")):
            with self.subTest(data=data[:10]), self.assertRaises(AppError):
                extract_pdf(data, "notes.pdf", Config())
        with self.assertRaises(AppError):
            extract_pdf(pdf_bytes("hello"), "notes.pdf", Config(max_file_bytes=10))

    def test_filename(self):
        self.assertEqual(safe_filename("../../a.pdf"), "a.pdf")
        self.assertEqual(safe_filename("C:\\docs\\a.pdf"), "a.pdf")

    def test_encrypted(self):
        with pymupdf.open(stream=pdf_bytes("secret"), filetype="pdf") as pdf:
            data = pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="secret")
        with self.assertRaisesRegex(AppError, "password"):
            extract_pdf(data, "encrypted.pdf", Config())
