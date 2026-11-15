"""
Automated evaluation test suite with 12 benchmark questions covering:
1. Direct fact lookups
2. Multi-turn conversational memory follow-ups
3. Cross-document synthesis
4. Out-of-scope anti-hallucination fallback triggers
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import SAMPLE_REPORTS_DIR
from app.ingestion import process_pdf_document
from app.vectorstore import FAISSVectorStoreManager, LocalDenseEmbeddings
from app.memory import ConversationMemory
from app.graph import build_docuchat_graph


EVALUATION_QUESTIONS = [
    # 1. Direct Fact Lookups
    {
        "id": "Q01",
        "category": "Direct Fact Lookup",
        "session": "eval_fact_1",
        "question": "What was TechCorp total revenue in Q3 2024 and what was the year-over-year growth rate?",
        "expected_doc": "TechCorp_Q3_2024_Financial_Report.pdf",
        "must_contain": ["42.8", "14%"],
        "expect_fallback": False
    },
    {
        "id": "Q02",
        "category": "Direct Fact Lookup",
        "session": "eval_fact_2",
        "question": "What is the SLA response time for a P1 critical incident and who is the escalation lead?",
        "expected_doc": "Cloud_Operations_Incident_Runbook.pdf",
        "must_contain": ["5 Minutes", "Incident Commander"],
        "expect_fallback": False
    },
    {
        "id": "Q03",
        "category": "Direct Fact Lookup",
        "session": "eval_fact_3",
        "question": "Under GDPR and CCPA, what is the mandatory resolution SLA for Subject Access Requests (SAR)?",
        "expected_doc": "Global_Customer_Data_Privacy_Standard.pdf",
        "must_contain": ["30", "day"],
        "expect_fallback": False
    },
    {
        "id": "Q04",
        "category": "Direct Fact Lookup",
        "session": "eval_fact_4",
        "question": "What minimum harmlessness benchmark score must AI models achieve before production deployment?",
        "expected_doc": "Enterprise_AI_Governance_Policy.pdf",
        "must_contain": ["98.5%", "harmlessness"],
        "expect_fallback": False
    },

    # 2. Multi-Turn Conversational Memory & Coreference Resolution
    {
        "id": "Q05",
        "category": "Multi-Turn Memory (Turn 1)",
        "session": "eval_multiturn_chain",
        "question": "Tell me about TechCorp Cloud and Hyperscale Infrastructure performance in Q3 2024.",
        "expected_doc": "TechCorp_Q3_2024_Financial_Report.pdf",
        "must_contain": ["18.4"],
        "expect_fallback": False
    },
    {
        "id": "Q06",
        "category": "Multi-Turn Memory (Turn 2)",
        "session": "eval_multiturn_chain",
        "question": "How does that compare to Generative AI and Data Services revenue?",
        "expected_doc": "TechCorp_Q3_2024_Financial_Report.pdf",
        "must_contain": ["Generative AI"],
        "expect_fallback": False
    },
    {
        "id": "Q07",
        "category": "Multi-Turn Memory (Turn 3)",
        "session": "eval_multiturn_chain",
        "question": "What capital investments are planned to support this demand?",
        "expected_doc": "TechCorp_Q3_2024_Financial_Report.pdf",
        "must_contain": ["8.5", "Titan TPU"],
        "expect_fallback": False
    },

    # 3. Cross-Document Synthesis
    {
        "id": "Q08",
        "category": "Cross-Document Synthesis",
        "session": "eval_cross_1",
        "question": "How do AI model testing requirements in our governance policy govern deployments on our cloud infrastructure?",
        "expected_doc": "Enterprise_AI_Governance_Policy.pdf",
        "must_contain": ["98.5%"],
        "expect_fallback": False
    },
    {
        "id": "Q09",
        "category": "Cross-Document Synthesis",
        "session": "eval_cross_2",
        "question": "What is the mandatory 72-hour breach notification requirement under customer data privacy standards and how must PII be encrypted?",
        "expected_doc": "Global_Customer_Data_Privacy_Standard.pdf",
        "must_contain": ["72 hours", "AES-256"],
        "expect_fallback": False
    },

    # 4. Out-of-Scope Negative Questions (Anti-Hallucination Fallback)
    {
        "id": "Q10",
        "category": "Anti-Hallucination Fallback",
        "session": "eval_fallback_1",
        "question": "What is the employee health insurance reimbursement limit for overseas dental surgery?",
        "expected_doc": None,
        "must_contain": ["couldn't find"],
        "expect_fallback": True
    },
    {
        "id": "Q11",
        "category": "Anti-Hallucination Fallback",
        "session": "eval_fallback_2",
        "question": "What is the annual paid vacation allowance for software engineers in Europe?",
        "expected_doc": None,
        "must_contain": ["couldn't find"],
        "expect_fallback": True
    },
    {
        "id": "Q12",
        "category": "Anti-Hallucination Fallback",
        "session": "eval_fallback_3",
        "question": "What was the CEO personal stock option compensation payout for 2024?",
        "expected_doc": None,
        "must_contain": ["couldn't find"],
        "expect_fallback": True
    }
]


def run_evaluations():
    print("=" * 85)
    print("DOCUCHAT: AUTOMATED EVALUATION SUITE (12 BENCHMARK QUESTIONS)")
    print("=" * 85)

    # Setup Vector Store with all 4 sample PDFs
    vs = FAISSVectorStoreManager(embeddings=LocalDenseEmbeddings())
    vs.clear()

    print("\n[1/3] Ingesting all sample reports into FAISS index...")
    for pdf_file in SAMPLE_REPORTS_DIR.glob("*.pdf"):
        chunks, stats = process_pdf_document(pdf_file)
        vs.add_documents(chunks)
        print(f"  ✓ {pdf_file.name:<45} | Chunks: {stats['total_chunks']}")

    memory = ConversationMemory()
    graph = build_docuchat_graph(vs)

    print("\n[2/3] Executing evaluation test questions...")
    results = []

    for item in EVALUATION_QUESTIONS:
        t0 = time.time()
        session_id = item["session"]
        history = memory.get_langchain_messages(session_id)

        state = {
            "question": item["question"],
            "session_id": session_id,
            "chat_history": history,
            "standalone_question": item["question"],
            "documents": [],
            "sources": [],
            "is_relevant": False,
            "answer": "",
            "doc_filter": None
        }

        output = graph.invoke(state)
        latency = time.time() - t0

        # Save to memory for multi-turn testing
        memory.add_message(session_id, "user", item["question"])
        memory.add_message(session_id, "assistant", output["answer"], sources=output["sources"])

        answer = output["answer"]
        sources = output["sources"]
        standalone = output["standalone_question"]

        # Content verification
        matched_terms = [t for t in item["must_contain"] if t.lower() in answer.lower()]
        content_ok = len(matched_terms) == len(item["must_contain"])

        # Source verification
        if item["expect_fallback"]:
            source_ok = len(sources) == 0 or "couldn't find" in answer.lower()
        else:
            source_ok = any(item["expected_doc"].lower() in s["filename"].lower() for s in sources) if item["expected_doc"] else True

        passed = content_ok and source_ok
        verdict_str = "PASS ✓" if passed else "FAIL ✗"

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "standalone": standalone,
            "answer": answer,
            "sources": len(sources),
            "latency": latency,
            "passed": passed
        })

        print(f"  [{item['id']}] {item['category']:<32} | {verdict_str} ({latency:.2f}s)")
        if not passed:
            print(f"       Expected terms: {item['must_contain']} | Found: {matched_terms}")
            print(f"       Answer excerpt: {answer[:140]}...")

    # Summary Table
    print("\n" + "=" * 85)
    print(f"{'ID':<4} | {'Category':<30} | {'Sources':<8} | {'Latency':<8} | {'Status':<6}")
    print("-" * 85)
    passed_count = sum(1 for r in results if r["passed"])
    for r in results:
        status_tag = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:<4} | {r['category']:<30} | {r['sources']:<8} | {r['latency']:<7.2f}s | {status_tag:<6}")

    print("-" * 85)
    score = (passed_count / len(results)) * 100
    print(f"Evaluation Score: {passed_count}/{len(results)} ({score:.1f}%)")
    print("=" * 85)

    return passed_count == len(results)


if __name__ == "__main__":
    success = run_evaluations()
    sys.exit(0 if success else 1)
