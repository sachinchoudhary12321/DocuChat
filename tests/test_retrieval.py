"""
Unit tests for FAISS Vector Store manager, similarity search, and persistence.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from app.vectorstore import FAISSVectorStoreManager, LocalDenseEmbeddings


def get_sample_chunks():
    return [
        Document(
            page_content="TechCorp Q3 2024 revenue reached $42.8 billion with Cloud delivering $18.4 billion.",
            metadata={"doc_id": "d1", "filename": "TechCorp_Q3_2024.pdf", "page": 1, "chunk_id": "c1"}
        ),
        Document(
            page_content="Mandatory 72-hour breach reporting window to data protection authorities under GDPR.",
            metadata={"doc_id": "d2", "filename": "Privacy_Standard.pdf", "page": 1, "chunk_id": "c2"}
        ),
        Document(
            page_content="P1 critical incidents require a 5-minute response SLA led by the Incident Commander.",
            metadata={"doc_id": "d3", "filename": "Incident_Runbook.pdf", "page": 1, "chunk_id": "c3"}
        )
    ]


@pytest.fixture
def sample_chunks():
    return get_sample_chunks()


def test_vector_store_crud(tmp_path, sample_chunks):
    embedder = LocalDenseEmbeddings()
    vs = FAISSVectorStoreManager(persist_dir=tmp_path, embeddings=embedder)

    # 1. Add chunks
    added = vs.add_documents(sample_chunks)
    assert added == 3
    assert len(vs.documents) == 3

    # 2. Search
    results = vs.similarity_search_with_score("What was TechCorp revenue?", k=2)
    assert len(results) > 0
    top_doc, score = results[0]
    assert top_doc.metadata["filename"] == "TechCorp_Q3_2024.pdf"
    assert "42.8" in top_doc.page_content
    assert score > 0.0

    # 3. Document Filter Search
    filtered = vs.similarity_search_with_score("incident response", k=2, doc_filter="Incident_Runbook.pdf")
    assert len(filtered) == 1
    assert filtered[0][0].metadata["filename"] == "Incident_Runbook.pdf"

    # 4. Persistence Reload Test
    vs_reloaded = FAISSVectorStoreManager(persist_dir=tmp_path, embeddings=embedder)
    assert len(vs_reloaded.documents) == 3

    # 5. Delete Document
    deleted = vs.delete_document("Privacy_Standard.pdf")
    assert deleted == 1
    assert len(vs.documents) == 2

    # 6. Clear
    vs.clear()
    assert len(vs.documents) == 0
