"""
Document ingestion pipeline for DocuChat.
Handles PDF validation, page-level text extraction, text sanitization,
recursive character chunking, and metadata enrichment.
"""

import os
import re
import uuid
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any
from pypdf import PdfReader
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings


def compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file for integrity and deduplication."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_text(text: str) -> str:
    """Sanitize extracted PDF text while preserving sentence structure."""
    if not text:
        return ""
    # Strip null bytes and non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize multiple whitespace while keeping paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def validate_pdf_file(file_path: str | Path, max_size_mb: int = None) -> Path:
    """Validate that the file exists, is a valid PDF, and conforms to size limits."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid file format: '{path.name}'. Only PDF files are supported.")

    limit_mb = max_size_mb or settings.MAX_UPLOAD_SIZE_MB
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > limit_mb:
        raise ValueError(
            f"File '{path.name}' exceeds the maximum allowed size of {limit_mb} MB (actual: {file_size_mb:.2f} MB)."
        )

    return path


def extract_pages_from_pdf(file_path: str | Path) -> List[Document]:
    """
    Extract text page-by-page from a PDF file.
    Raises ValueError if PDF is scanned, empty, or unreadable.
    """
    path = validate_pdf_file(file_path)
    file_hash = compute_file_hash(path)
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{path.name}:{file_hash}"))[:12]

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse PDF '{path.name}': corrupted or unreadable format. Details: {e}")

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"PDF file '{path.name}' contains 0 pages.")

    documents: List[Document] = []
    total_extracted_chars = 0

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        if cleaned:
            total_extracted_chars += len(cleaned)
            doc = Document(
                page_content=cleaned,
                metadata={
                    "doc_id": doc_id,
                    "filename": path.name,
                    "file_path": str(path.resolve()),
                    "page": page_idx + 1,  # 1-indexed for citation
                    "total_pages": total_pages,
                    "doc_hash": file_hash
                }
            )
            documents.append(doc)

    # Scanned PDF / OCR Guardrail: Check if readable text was extracted
    if total_extracted_chars < 50:
        raise ValueError(
            f"PDF '{path.name}' appears to be a scanned image or empty. "
            "No extractable text was found. Please upload a PDF with embedded text."
        )

    return documents


def split_into_chunks(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    """
    Split page documents into overlapping chunks with metadata tracking.
    Preserves document name, page number, and injects unique chunk_id.
    """
    c_size = chunk_size or settings.CHUNK_SIZE
    c_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len
    )

    split_chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(split_chunks):
        doc_id = chunk.metadata.get("doc_id", "doc")
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_id"] = f"{doc_id}_p{page}_c{idx}"
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["char_count"] = len(chunk.page_content)

    return split_chunks


def process_pdf_document(
    file_path: str | Path,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Full ingestion pipeline for a single PDF:
    Validation -> Page Extraction -> Chunking -> Metadata Attachment -> Summary Stats.
    """
    raw_pages = extract_pages_from_pdf(file_path)
    chunks = split_into_chunks(raw_pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    path = Path(file_path)
    stats = {
        "doc_id": raw_pages[0].metadata.get("doc_id", "unknown") if raw_pages else "",
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "total_pages": len(raw_pages),
        "total_chunks": len(chunks),
        "total_characters": sum(len(c.page_content) for c in chunks),
        "avg_chunk_size": int(sum(len(c.page_content) for c in chunks) / max(len(chunks), 1))
    }

    return chunks, stats
