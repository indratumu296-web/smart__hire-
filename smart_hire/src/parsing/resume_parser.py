from __future__ import annotations

from io import BytesIO


def extract_resume_text(uploaded_file) -> str:
    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]
    payload = uploaded_file.getvalue()

    if suffix == "txt":
        return payload.decode("utf-8", errors="ignore")
    if suffix == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise ValueError("PDF support needs pypdf. Install the packages in requirements.txt.") from error
        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)
    if suffix == "docx":
        try:
            from docx import Document
        except ImportError as error:
            raise ValueError("DOCX support needs python-docx. Install the packages in requirements.txt.") from error
        document = Document(BytesIO(payload))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise ValueError("Unsupported file type. Upload a PDF, DOCX, or TXT resume.")
