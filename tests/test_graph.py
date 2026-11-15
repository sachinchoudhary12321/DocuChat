"""
Unit tests for LangGraph flow nodes:
rewrite_question, retrieve, grade_relevance, generate, and fallback.
"""

import pytest
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

from app.vectorstore import FAISSVectorStoreManager, LocalDenseEmbeddings
from app.ingestion import process_pdf_document
from app.graph import build_docuchat_graph, LocalDeterministicLLM
from app.config import SAMPLE_REPORTS_DIR


def get_populated_vector_store(tmp_path):
    vs = FAISSVectorStoreManager(persist_dir=tmp_path, embeddings=LocalDenseEmbeddings())
    # Index sample financial report and runbook
    for name in ["TechCorp_Q3_2024_Financial_Report.pdf", "Cloud_Operations_Incident_Runbook.pdf"]:
        pdf_path = SAMPLE_REPORTS_DIR / name
        chunks, _ = process_pdf_document(pdf_path)
        vs.add_documents(chunks)
    return vs


@pytest.fixture
def populated_vector_store(tmp_path):
    return get_populated_vector_store(tmp_path)


def test_question_rewriter_resolves_pronoun(populated_vector_store):
    graph = build_docuchat_graph(populated_vector_store)

    chat_history = [
        HumanMessage(content="Tell me about TechCorp Cloud division revenue."),
        AIMessage(content="Cloud division generated $18.4 billion.")
    ]

    initial_state = {
        "question": "How does that compare to the previous year?",
        "session_id": "test_sess",
        "chat_history": chat_history,
        "standalone_question": "",
        "documents": [],
        "sources": [],
        "is_relevant": False,
        "answer": "",
        "doc_filter": None
    }

    result = graph.invoke(initial_state)
    standalone = result["standalone_question"]
    # Check that ambiguous pronoun "that" was rewritten into standalone form
    assert "TechCorp" in standalone or "Cloud" in standalone or "revenue" in standalone
    assert "that" not in standalone.lower().split()


def test_direct_fact_generation_with_citation(populated_vector_store):
    graph = build_docuchat_graph(populated_vector_store)

    initial_state = {
        "question": "What was TechCorp total revenue in Q3 2024?",
        "session_id": "test_fact",
        "chat_history": [],
        "standalone_question": "What was TechCorp total revenue in Q3 2024?",
        "documents": [],
        "sources": [],
        "is_relevant": False,
        "answer": "",
        "doc_filter": None
    }

    result = graph.invoke(initial_state)
    assert result["is_relevant"] is True
    assert "42.8" in result["answer"]
    assert len(result["sources"]) > 0
    # Verify inline citation pattern [Doc, p. X]
    assert "[TechCorp_Q3_2024_Financial_Report.pdf, p. 1]" in result["answer"]


def test_out_of_scope_triggers_fallback(populated_vector_store):
    graph = build_docuchat_graph(populated_vector_store)

    initial_state = {
        "question": "What is the corporate health insurance policy for dental implants?",
        "session_id": "test_negative",
        "chat_history": [],
        "standalone_question": "What is the corporate health insurance policy for dental implants?",
        "documents": [],
        "sources": [],
        "is_relevant": False,
        "answer": "",
        "doc_filter": None
    }

    result = graph.invoke(initial_state)
    # Must route to fallback and refuse cleanly
    assert "couldn't find" in result["answer"].lower()
    assert result["sources"] == []
