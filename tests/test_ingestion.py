"""
Unit tests for PDF validation, extraction, chunking, and error handling.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from app.ingestion import (
    validate_pdf_file,
    extract_pages_from_pdf,
    split_into_chunks,
    process_pdf_document,
    clean_text
)
from app.config import SAMPLE_REPORTS_DIR


def test_clean_text():
    raw = "Hello \x00 world!  \t This is   a test.\n\n\nNew paragraph."
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "  " not in cleaned
    assert "New paragraph." in cleaned


def test_validate_pdf_file_valid():
    pdf_path = SAMPLE_REPORTS_DIR / "TechCorp_Q3_2024_Financial_Report.pdf"
    validated = validate_pdf_file(pdf_path)
    assert validated.exists()
    assert validated.suffix.lower() == ".pdf"


def test_validate_pdf_file_invalid_extension(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello world")
    with pytest.raises(ValueError, match="Only PDF files are supported"):
        validate_pdf_file(txt_file)


def test_extract_pages_from_pdf():
    pdf_path = SAMPLE_REPORTS_DIR / "Cloud_Operations_Incident_Runbook.pdf"
    pages = extract_pages_from_pdf(pdf_path)
    assert len(pages) >= 1
    assert pages[0].metadata["page"] == 1
    assert "doc_id" in pages[0].metadata
    assert pages[0].metadata["filename"] == "Cloud_Operations_Incident_Runbook.pdf"
    assert "SLA" in pages[0].page_content


def test_split_into_chunks():
    doc = Document(
        page_content="This is a sample document text that repeats. " * 30,
        metadata={"doc_id": "test_id", "filename": "sample.pdf", "page": 1}
    )
    chunks = split_into_chunks([doc], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert "chunk_id" in c.metadata
        assert "chunk_index" in c.metadata
        assert c.metadata["page"] == 1
        assert len(c.page_content) <= 350


def test_process_pdf_document_stats():
    pdf_path = SAMPLE_REPORTS_DIR / "Global_Customer_Data_Privacy_Standard.pdf"
    chunks, stats = process_pdf_document(pdf_path)
    assert stats["filename"] == "Global_Customer_Data_Privacy_Standard.pdf"
    assert stats["total_chunks"] == len(chunks)
    assert stats["total_pages"] >= 1
    assert stats["total_characters"] > 0
