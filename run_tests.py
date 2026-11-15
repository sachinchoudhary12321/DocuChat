"""
Master test runner for DocuChat.
Runs unit tests for ingestion, vector store, LangGraph, and evaluation benchmark.
"""

import sys
import traceback
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import shutil

from tests.test_ingestion import (
    test_clean_text,
    test_validate_pdf_file_valid,
    test_validate_pdf_file_invalid_extension,
    test_extract_pages_from_pdf,
    test_split_into_chunks,
    test_process_pdf_document_stats
)
from tests.test_retrieval import (
    test_vector_store_crud,
    get_sample_chunks
)
from tests.test_graph import (
    test_question_rewriter_resolves_pronoun,
    test_direct_fact_generation_with_citation,
    test_out_of_scope_triggers_fallback,
    get_populated_vector_store
)
from tests.test_eval_questions import run_evaluations


def run_all_docuchat_tests():
    print("=" * 80)
    print("DOCUCHAT: FULL TEST SUITE EXECUTION")
    print("=" * 80)

    unit_tests = [
        # Ingestion tests
        ("test_clean_text", lambda: test_clean_text()),
        ("test_validate_pdf_file_valid", lambda: test_validate_pdf_file_valid()),
        ("test_validate_pdf_file_invalid_extension", lambda: run_with_tmp(test_validate_pdf_file_invalid_extension)),
        ("test_extract_pages_from_pdf", lambda: test_extract_pages_from_pdf()),
        ("test_split_into_chunks", lambda: test_split_into_chunks()),
        ("test_process_pdf_document_stats", lambda: test_process_pdf_document_stats()),

        # Retrieval tests
        ("test_vector_store_crud", lambda: run_retrieval_crud()),

        # LangGraph flow tests
        ("test_question_rewriter_resolves_pronoun", lambda: run_graph_test(test_question_rewriter_resolves_pronoun)),
        ("test_direct_fact_generation_with_citation", lambda: run_graph_test(test_direct_fact_generation_with_citation)),
        ("test_out_of_scope_triggers_fallback", lambda: run_graph_test(test_out_of_scope_triggers_fallback)),
    ]

    passed = 0
    failed = 0

    print("\n[PART 1: Unit & Component Tests]")
    for name, func in unit_tests:
        try:
            print(f"  Running {name:<45} ... ", end="", flush=True)
            func()
            print("[PASS]")
            passed += 1
        except Exception as e:
            print("[FAIL]")
            traceback.print_exc()
            failed += 1

    print(f"\nUnit Tests Summary: {passed}/{len(unit_tests)} Passed.")

    print("\n[PART 2: Benchmark Evaluation Suite (12 Questions)]")
    eval_ok = run_evaluations()

    print("\n" + "=" * 80)
    total_ok = (failed == 0) and eval_ok
    status = "SUCCESSFUL (ALL TESTS PASSED)" if total_ok else "FAILED"
    print(f"OVERALL STATUS: {status}")
    print("=" * 80)
    return total_ok


def run_with_tmp(fn):
    tmp = Path(tempfile.mkdtemp())
    try:
        fn(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_retrieval_crud():
    tmp = Path(tempfile.mkdtemp())
    try:
        chunks = get_sample_chunks()
        test_vector_store_crud(tmp, chunks)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_graph_test(fn):
    tmp = Path(tempfile.mkdtemp())
    try:
        vs = get_populated_vector_store(tmp)
        fn(vs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    success = run_all_docuchat_tests()
    sys.exit(0 if success else 1)
