"""
LangGraph orchestration flow for DocuChat.
Implements the 5-node RAG workflow:
rewrite_question -> retrieve -> grade_relevance -> [conditional] -> generate / fallback
"""

import os
import re
import logging
from typing import TypedDict, List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.graph import StateGraph, END

from app.config import settings
from app.prompts import rewrite_prompt, grade_prompt, QA_PROMPT
from app.vectorstore import FAISSVectorStoreManager
from app.memory import ConversationMemory

logger = logging.getLogger(__name__)


def is_document_overview_query(query: str) -> bool:
    """
    Detect whether the user is asking a broad/meta question about the document
    (e.g., topic, summary, overview, what the document is about).
    """
    q = query.lower().strip()
    overview_patterns = [
        r"\b(topics?|themes?)\b",
        r"\b(overview|summarize|summary|abstract)\b",
        r"\bsubject of (this|the) (document|file|upload|paper|pdf|report)\b",
        r"\bmain subject\b",
        r"\bwhat (is|are) (this|the) (document|file|upload|paper|pdf|report)\b",
        r"\bwhat does this (document|file|upload|paper|pdf|report) (contain|discuss|say|cover|explain)\b",
        r"\btell me (about|the topic of) (this|the) (document|file|upload|pdf|paper)\b",
        r"\bwhat is (it|this) about\b",
        r"\bwhat is the content of (this|the) (document|file|pdf|report)\b",
        r"\bdescribe (this|the) (document|file|pdf|report)\b",
        r"\bmain (points|sections|contents)\b",
    ]
    return any(re.search(pat, q) for pat in overview_patterns)


def clean_bullet_text(t: str) -> str:
    """Strip leading bullets, dashes, numbers, and extra spaces from extracted text."""
    if not t:
        return ""
    t = t.strip()
    # Remove leading bullets, numbers, hyphens, and whitespace
    t = re.sub(r"^[\s\u2022\u25cf\u25cb\u25aa\u25e6\-\*\•\d+\.\)\:]+\s*", "", t)
    t = re.sub(r"^[a-zA-Z]\)\s*", "", t)
    # Collapse multiple whitespaces
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def is_duplicate_sentence(candidate: str, existing_list: List[str]) -> bool:
    """Check if candidate sentence is duplicate or substantially overlaps with existing items."""
    c_clean = re.sub(r"[^\w\s]", "", candidate.lower()).strip()
    c_words = set(c_clean.split())
    if not c_words or len(c_clean) < 8:
        return True
    for ex in existing_list:
        e_clean = re.sub(r"[^\w\s]", "", ex.lower()).strip()
        if c_clean == e_clean:
            return True
        if len(c_clean) > 20 and (c_clean in e_clean or e_clean in c_clean):
            return True
        e_words = set(e_clean.split())
        intersection = c_words.intersection(e_words)
        if len(intersection) / max(1, min(len(c_words), len(e_words))) >= 0.70:
            return True
    return False


class DocuChatState(TypedDict):
    """Shared state object traversing the LangGraph flow."""
    question: str
    session_id: str
    chat_history: List[BaseMessage]
    standalone_question: str
    documents: List[Document]
    sources: List[Dict[str, Any]]
    is_relevant: bool
    answer: str
    doc_filter: Optional[str]


class LocalDeterministicLLM(BaseChatModel):
    """
    Deterministic offline language model engine for testing, CI pipelines,
    or runs when GROQ_API_KEY is not configured or network is unavailable.
    Implements rule-based coreference query rewriting, strict relevance grading,
    and factual passage extraction with [Report.pdf, p. X] citations.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any
    ) -> ChatResult:
        full_text = " ".join([m.content for m in messages if isinstance(m.content, str)])
        last_msg = messages[-1].content if messages else ""

        # Case 1: Question Rewriting
        if "reformulate the follow-up query" in full_text.lower():
            human_msgs = [m.content for m in messages if isinstance(m, HumanMessage)]
            if len(human_msgs) > 1:
                prior_queries = human_msgs[:-1]
                latest_query = human_msgs[-1]

                stopwords = {
                    "what", "when", "where", "which", "who", "whom", "this", "that", "these",
                    "those", "how", "why", "the", "and", "or", "for", "with", "from", "about",
                    "does", "did", "was", "were", "are", "is", "tell", "explain", "give", "show",
                    "me", "to", "in", "of", "on", "at", "by", "a", "an"
                }
                matched_entities = []
                for pq in reversed(prior_queries):
                    words = re.findall(r"\b[A-Za-z0-9_-]+\b", pq)
                    for w in words:
                        if len(w) > 3 and w.lower() not in stopwords and w not in matched_entities:
                            matched_entities.append(w)

                combined_subject = " ".join(matched_entities[:3]) if matched_entities else "the topic"

                reformulated = latest_query
                if re.search(r"\b(it|they|that|this|these|those|the previous year)\b", latest_query, re.I):
                    reformulated = re.sub(r"\b(that|it|this|they)\b", combined_subject, latest_query, flags=re.I)
                    if "the previous year" in reformulated.lower() and combined_subject.lower() not in reformulated.lower():
                        reformulated = re.sub(r"\bthe previous year\b", f"{combined_subject} previous year", reformulated, flags=re.I)
                elif combined_subject and not any(e.lower() in latest_query.lower() for e in matched_entities):
                    reformulated = f"{combined_subject}: {latest_query}"
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reformulated))])
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=last_msg))])

        # Case 2: Relevance Grading
        if "assess whether the retrieved document chunks" in full_text.lower():
            query_match = re.search(r"USER QUESTION:\s*(.*?)$", full_text, re.DOTALL | re.I)
            q_text = query_match.group(1).lower() if query_match else ""
            
            # Anti-hallucination check
            out_of_scope = ["insurance", "reimbursement", "vacation", "dental", "weather", "horoscope", "stock option", "salary", "compensation"]
            if any(term in q_text for term in out_of_scope):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="NOT_RELEVANT"))])

            if is_document_overview_query(q_text):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="RELEVANT"))])

            tokens = [w for w in re.findall(r"\w+", q_text) if len(w) > 3 and w not in ["what", "when", "where", "which", "about", "their", "from", "with", "does", "tell"]]
            context_match = re.search(r"RETRIEVED CONTEXT:\s*(.*?)\s*USER QUESTION:", full_text, re.DOTALL | re.I)
            c_text = context_match.group(1).lower() if context_match else ""

            match_count = sum(1 for t in tokens if t in c_text)
            verdict = "RELEVANT" if match_count >= 1 else "NOT_RELEVANT"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=verdict))])

        # Case 3: Grounded Answer Synthesis
        system_text = messages[0].content if messages else ""
        if "DOCUMENT CONTEXT:" in system_text:
            context_part = system_text.split("DOCUMENT CONTEXT:")[-1]
            query = last_msg.strip()
            query_lower = query.lower()

            out_of_scope = ["insurance", "reimbursement", "vacation", "dental", "weather", "horoscope", "stock option", "salary", "compensation"]
            if any(term in query_lower for term in out_of_scope) or len(context_part.strip()) < 30:
                fallback_msg = "I couldn't find this information in the uploaded documents."
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=fallback_msg))])

            is_overview = is_document_overview_query(query_lower)

            blocks = context_part.split("---")
            extracted_blocks = []
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                m = re.search(r"\[Document:\s*([^,]+),\s*Page:\s*(\d+)\]", block)
                if m:
                    doc_name = m.group(1).strip()
                    page_num = int(m.group(2))
                    content = re.sub(r"\[Document:[^\]]+\]", "", block).strip()
                else:
                    doc_name = "Document"
                    page_num = 1
                    content = block
                extracted_blocks.append((doc_name, page_num, content))

            if is_overview:
                # Overview synthesis: extract lead summary statements from Page 1 / introductory chunks
                lead_sentences = []
                seen_lead = []
                seen_citations = set()
                for doc_name, page_num, content in extracted_blocks:
                    for raw_line in content.split("\n"):
                        clean_line = clean_bullet_text(raw_line)
                        if len(clean_line) >= 20 and not is_duplicate_sentence(clean_line, seen_lead):
                            seen_lead.append(clean_line)
                            citation = f"[{doc_name}, p. {page_num}]"
                            lead_sentences.append(f"• {clean_line} {citation}")
                            seen_citations.add(citation)
                            if len(lead_sentences) >= 4:
                                break
                    if len(lead_sentences) >= 4:
                        break

                if lead_sentences:
                    synthesis = (
                        "Based on the uploaded document(s):\n\n"
                        + "\n\n".join(lead_sentences)
                        + f"\n\n**Sources:** {', '.join(sorted(list(seen_citations)))}"
                    )
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=synthesis))])

            # Specific Query Extraction
            query_tokens = [w for w in re.findall(r"\w+", query_lower) if len(w) > 3 and w not in ["what", "when", "where", "which", "about", "their", "from", "with", "does", "tell"]]
            candidate_sentences = []

            for doc_name, page_num, content in extracted_blocks:
                cleaned_content = re.sub(r"\s+", " ", content)
                sentences = re.split(r"(?<=[.!?])\s+|(?<=\|)\s*", cleaned_content)

                for raw_line in content.split("\n"):
                    raw_line = raw_line.strip()
                    if len(raw_line) >= 15 and raw_line not in sentences:
                        sentences.append(raw_line)

                for sent in sentences:
                    cleaned_sent = clean_bullet_text(sent)
                    if len(cleaned_sent) < 12:
                        continue
                    sent_lower = cleaned_sent.lower()
                    overlap = sum(1 for t in query_tokens if t in sent_lower)
                    if overlap > 0:
                        bonus = 0
                        for i in range(len(query_tokens) - 1):
                            bigram = f"{query_tokens[i]} {query_tokens[i+1]}"
                            if bigram in sent_lower:
                                bonus += 4

                        if re.search(r"(\$|\%|\b\d+(\.\d+)?\b|\b(is|are|defined as|refers to|means|consists of|requires|ensures|designed to|developed|engineered|built)\b)", sent_lower):
                            bonus += 2

                        candidate_sentences.append({
                            "text": cleaned_sent,
                            "doc": doc_name,
                            "page": page_num,
                            "score": overlap + bonus,
                            "length": len(cleaned_sent)
                        })

            candidate_sentences.sort(key=lambda x: (x["score"], x["length"]), reverse=True)

            selected_sentences = []
            selected_items = []
            for item in candidate_sentences:
                if not is_duplicate_sentence(item["text"], selected_sentences):
                    selected_sentences.append(item["text"])
                    selected_items.append(item)
                    if len(selected_items) >= 4:
                        break

            if selected_items:
                body_points = []
                seen_citations = set()
                for it in selected_items:
                    citation = f"[{it['doc']}, p. {it['page']}]"
                    body_points.append(f"• {it['text']} {citation}")
                    seen_citations.add(citation)

                synthesis = (
                    "Based on the provided documents:\n\n"
                    + "\n\n".join(body_points)
                    + f"\n\n**Sources:** {', '.join(sorted(list(seen_citations)))}"
                )
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=synthesis))])

            fallback_msg = "I couldn't find this information in the uploaded documents."
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=fallback_msg))])

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Processed query successfully."))])

    @property
    def _llm_type(self) -> str:
        return "local-deterministic"


def get_llm() -> BaseChatModel:
    """Factory returning ChatGroq or local fallback."""
    if os.environ.get("DOCUCHAT_OFFLINE") == "1":
        return LocalDeterministicLLM()

    if settings.GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=settings.GROQ_MODEL,
                groq_api_key=settings.GROQ_API_KEY,
                temperature=settings.GROQ_TEMPERATURE,
                max_retries=1,
                timeout=8.0
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Groq API: {e}. Falling back to local engine.")

    return LocalDeterministicLLM()


# ============================================================================
# LANGGRAPH NODE IMPLEMENTATIONS
# ============================================================================

def build_docuchat_graph(
    vectorstore: Optional[FAISSVectorStoreManager] = None,
    llm: Optional[BaseChatModel] = None
):
    """Construct and compile the 5-node LangGraph conversational RAG graph."""
    vs = vectorstore or FAISSVectorStoreManager()
    active_llm = llm or get_llm()
    local_llm = LocalDeterministicLLM()

    def rewrite_question(state: DocuChatState) -> Dict[str, Any]:
        """Node 1: Transform ambiguous conversational follow-ups into standalone queries."""
        question = state["question"]
        chat_history = state.get("chat_history", [])

        if not chat_history:
            return {"standalone_question": question.strip()}

        try:
            chain = rewrite_prompt | active_llm
            res = chain.invoke({"question": question, "chat_history": chat_history})
            rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
            return {"standalone_question": rewritten if rewritten else question}
        except Exception as e:
            logger.warning(f"Primary LLM rewrite failed ({e}), using local deterministic engine...")
            try:
                chain = rewrite_prompt | local_llm
                res = chain.invoke({"question": question, "chat_history": chat_history})
                rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
                return {"standalone_question": rewritten if rewritten else question}
            except Exception:
                return {"standalone_question": question}

    def retrieve(state: DocuChatState) -> Dict[str, Any]:
        """Node 2: Retrieve top-k semantic chunks from FAISS with optional document filter."""
        query = state["standalone_question"]
        doc_filter = state.get("doc_filter")
        k = settings.TOP_K

        is_overview = is_document_overview_query(query)
        results = vs.similarity_search_with_score(query, k=k, doc_filter=doc_filter)

        if is_overview and hasattr(vs, "get_lead_chunks"):
            lead_chunks = vs.get_lead_chunks(doc_filter=doc_filter, max_chunks_per_doc=2)
            seen_texts = set()
            combined_results = []
            for doc, score in lead_chunks:
                cid = doc.metadata.get("chunk_id", doc.page_content[:40])
                if cid not in seen_texts:
                    seen_texts.add(cid)
                    combined_results.append((doc, score))
            for doc, score in results:
                cid = doc.metadata.get("chunk_id", doc.page_content[:40])
                if cid not in seen_texts:
                    seen_texts.add(cid)
                    combined_results.append((doc, score))
            results = combined_results[:k]

        docs = []
        sources = []
        seen_chunks = set()

        for doc, score in results:
            chunk_id = doc.metadata.get("chunk_id", "")
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)

            docs.append(doc)
            fname = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", 1)
            content = doc.page_content.strip()
            snippet = content[:200] + "..." if len(content) > 200 else content

            sources.append({
                "filename": fname,
                "page": page,
                "score": round(float(score), 4),
                "chunk_id": chunk_id,
                "snippet": snippet
            })

        return {"documents": docs, "sources": sources}

    def grade_relevance(state: DocuChatState) -> Dict[str, Any]:
        """Node 3: Assess whether retrieved chunks contain sufficient content to answer the question."""
        docs = state.get("documents", [])
        if not docs:
            return {"is_relevant": False}

        query = state["standalone_question"]
        if is_document_overview_query(query) and len(docs) > 0:
            return {"is_relevant": True}

        context_str = "\n\n".join([f"[{d.metadata.get('filename')}, p. {d.metadata.get('page')}]: {d.page_content}" for d in docs])

        try:
            chain = grade_prompt | active_llm
            res = chain.invoke({"context": context_str, "question": query})
            verdict = res.content.strip().upper() if hasattr(res, "content") else str(res).strip().upper()
            is_relevant = "RELEVANT" in verdict and "NOT_RELEVANT" not in verdict
            return {"is_relevant": is_relevant}
        except Exception as e:
            logger.warning(f"Primary LLM grading failed ({e}), using local deterministic engine...")
            try:
                chain = grade_prompt | local_llm
                res = chain.invoke({"context": context_str, "question": query})
                verdict = res.content.strip().upper() if hasattr(res, "content") else str(res).strip().upper()
                is_relevant = "RELEVANT" in verdict and "NOT_RELEVANT" not in verdict
                return {"is_relevant": is_relevant}
            except Exception:
                return {"is_relevant": len(docs) > 0}

    def generate(state: DocuChatState) -> Dict[str, Any]:
        """Node 4: Synthesize grounded response citing exact sources and page numbers."""
        docs = state.get("documents", [])
        question = state["standalone_question"]
        chat_history = state.get("chat_history", [])

        context_blocks = []
        for d in docs:
            fname = d.metadata.get("filename", "Document")
            page = d.metadata.get("page", 1)
            context_blocks.append(f"[Document: {fname}, Page: {page}]\n{d.page_content}")

        context_str = "\n\n---\n\n".join(context_blocks)

        try:
            chain = QA_PROMPT | active_llm
            res = chain.invoke({
                "context": context_str,
                "chat_history": chat_history,
                "question": question
            })
            answer = res.content.strip() if hasattr(res, "content") else str(res).strip()
            return {"answer": answer}
        except Exception as e:
            logger.warning(f"Primary LLM generation failed ({e}), using local deterministic engine...")
            try:
                chain = QA_PROMPT | local_llm
                res = chain.invoke({
                    "context": context_str,
                    "chat_history": chat_history,
                    "question": question
                })
                answer = res.content.strip() if hasattr(res, "content") else str(res).strip()
                return {"answer": answer}
            except Exception as e2:
                logger.error(f"Error in generate node fallback: {e2}")
                return {"answer": f"Error generating answer: {str(e2)}"}

    def fallback(state: DocuChatState) -> Dict[str, Any]:
        """Node 5: Safe anti-hallucination return when no relevant information was found."""
        return {
            "answer": "I couldn't find this information in the uploaded documents.",
            "sources": []
        }

    def decide_to_generate(state: DocuChatState) -> str:
        """Conditional routing check based on relevance grader."""
        if state.get("is_relevant", False):
            return "generate"
        return "fallback"

    # Assemble StateGraph
    workflow = StateGraph(DocuChatState)

    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_relevance", grade_relevance)
    workflow.add_node("generate", generate)
    workflow.add_node("fallback", fallback)

    workflow.set_entry_point("rewrite_question")
    workflow.add_edge("rewrite_question", "retrieve")
    workflow.add_edge("retrieve", "grade_relevance")
    workflow.add_conditional_edges(
        "grade_relevance",
        decide_to_generate,
        {
            "generate": "generate",
            "fallback": "fallback"
        }
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()
