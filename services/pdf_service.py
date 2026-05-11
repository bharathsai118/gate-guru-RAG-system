from __future__ import annotations

import re
from pathlib import Path


def clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed. Run pip install -r requirements.txt.") from exc

    document = None
    try:
        document = fitz.open(str(pdf_path))
        if document.is_encrypted:
            try:
                document.authenticate("")
            except Exception:
                raise ValueError("Encrypted PDFs are not supported.")

        pages: list[dict] = []
        for page_number, page in enumerate(document, start=1):
            text = clean_pdf_text(page.get_text("text") or "")
            if text:
                pages.append({"page_number": page_number, "text": text})

        if not pages:
            raise ValueError("No extractable text was found in this PDF.")

        return pages
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not extract text from PDF: {exc}") from exc
    finally:
        if document is not None:
            document.close()

