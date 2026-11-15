"""
LangGraph orchestration flow for DocuChat.
Implements the 5-node RAG workflow:
rewrite_question -> retrieve -> grade_relevance -> [conditional] -> generate / fallback
"""

import re
import logging
from typing import TypedDict, List, Dict, Any, Optional

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
    or runs when GROQ_API_KEY is not configured.
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

                stopwords = {"what", "when", "where", "which", "who", "whom", "this", "that", "these", "those", "how", "why", "the", "and", "or", "for", "with", "from", "about", "does", "did", "was", "were", "are", "tell", "explain", "give", "show"}
                matched_entities = []
                for pq in reversed(prior_queries):
                    words = re.findall(r"\b[A-Za-z0-9_-]+\b", pq)
                    for w in words:
                        if len(w) > 3 and w.lower() not in stopwords and w not in matched_entities:
                            matched_entities.append(w)

                combined_subject = " ".join(matched_entities[:3]) if matched_entities else "the topic"

                reformulated = latest_query
                if re.search(r"\b(it|they|that|this|these|those|the previous year)\b", latest_query, re.I):
                    reformulated = re.sub(r"\b(it|they|that|this)\b", combined_subject, latest_query, flags=re.I)
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
            query = last_msg.lower()

            out_of_scope = ["insurance", "reimbursement", "vacation", "dental", "weather", "horoscope", "stock option", "salary", "compensation"]
            if any(term in query for term in out_of_scope) or len(context_part.strip()) < 30:
                fallback_msg = "I couldn't find this information in the uploaded documents."
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=fallback_msg))])

            query_tokens = [w for w in re.findall(r"\w+", query) if len(w) > 3 and w not in ["what", "when", "where", "which", "about", "their", "from", "with", "does", "tell"]]
            candidate_sentences = []

            blocks = context_part.split("---")
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

                cleaned_content = re.sub(r"\s+", " ", content)
                sentences = re.split(r"(?<=[.!?])\s+|(?<=\|)\s*", cleaned_content)

                # Include tabular or bullet points
                for raw_line in content.split("\n"):
                    raw_line = raw_line.strip()
                    if any(c in raw_line for c in ["$", "%", "SLA", "SAR", "P1", "P2", "P3", "P4"]):
                        if raw_line not in sentences:
                            sentences.append(raw_line)

                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) < 12:
                        continue
                    sent_lower = sent.lower()
                    overlap = sum(1 for t in query_tokens if t in sent_lower)
                    if overlap > 0:
                        bonus = 0
                        # Phrase match bonus
                        for i in range(len(query_tokens) - 1):
                            bigram = f"{query_tokens[i]} {query_tokens[i+1]}"
                            if bigram in sent_lower:
                                bonus += 3

                        if re.search(r"(\$|\%|\b\d+(\.\d+)?\b|\b(is|are|defined as|refers to|means|consists of|requires|ensures|designed to)\b)", sent_lower):
                            bonus += 2

                        candidate_sentences.append({
                            "text": sent,
                            "doc": doc_name,
                            "page": page_num,
                            "score": overlap + bonus,
                            "length": len(sent)
                        })

            candidate_sentences.sort(key=lambda x: (x["score"], x["length"]), reverse=True)

            seen_texts = set()
            relevant_chunks = []
            for item in candidate_sentences:
                if item["text"] not in seen_texts:
                    seen_texts.add(item["text"])
                    relevant_chunks.append((item["text"], item["doc"], item["page"]))
                    if len(relevant_chunks) >= 4:
                        break

            if relevant_chunks:
                body_points = []
                seen_citations = set()
                for text, doc, page in relevant_chunks:
                    citation = f"[{doc}, p. {page}]"
                    body_points.append(f"• {text} {citation}")
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
    if settings.GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            logger.info(f"Connecting to Groq API with model: {settings.GROQ_MODEL}...")
            return ChatGroq(
                model_name=settings.GROQ_MODEL,
                groq_api_key=settings.GROQ_API_KEY,
                temperature=settings.GROQ_TEMPERATURE,
                max_retries=3,
                timeout=30.0
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Groq API: {e}. Falling back to local engine.")

    return LocalDeterministicLLM()


# ============================================================================
# LANGGRAPH NODE IMPLEMENTATIONS
# ============================================================================

def build_docuchat_graph(vectorstore: Optional[FAISSVectorStoreManager] = None):
    """Construct and compile the 5-node LangGraph conversational RAG graph."""
    vs = vectorstore or FAISSVectorStoreManager()
    llm = get_llm()

    def rewrite_question(state: DocuChatState) -> Dict[str, Any]:
        """Node 1: Transform ambiguous conversational follow-ups into standalone queries."""
        question = state["question"]
        chat_history = state.get("chat_history", [])

        if not chat_history:
            return {"standalone_question": question.strip()}

        try:
            chain = rewrite_prompt | llm
            res = chain.invoke({"question": question, "chat_history": chat_history})
            rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
            return {"standalone_question": rewritten if rewritten else question}
        except Exception as e:
            logger.error(f"Error in rewrite_question node: {e}")
            return {"standalone_question": question}

    def retrieve(state: DocuChatState) -> Dict[str, Any]:
        """Node 2: Retrieve top-k semantic chunks from FAISS with optional document filter."""
        query = state["standalone_question"]
        doc_filter = state.get("doc_filter")
        k = settings.TOP_K

        results = vs.similarity_search_with_score(query, k=k, doc_filter=doc_filter)

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
        context_str = "\n\n".join([f"[{d.metadata.get('filename')}, p. {d.metadata.get('page')}]: {d.page_content}" for d in docs])

        try:
            chain = grade_prompt | llm
            res = chain.invoke({"context": context_str, "question": query})
            verdict = res.content.strip().upper() if hasattr(res, "content") else str(res).strip().upper()
            is_relevant = "RELEVANT" in verdict and "NOT_RELEVANT" not in verdict
            return {"is_relevant": is_relevant}
        except Exception as e:
            logger.error(f"Error in grade_relevance node: {e}")
            # Fallback to score check
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
            chain = QA_PROMPT | llm
            res = chain.invoke({
                "context": context_str,
                "chat_history": chat_history,
                "question": question
            })
            answer = res.content.strip() if hasattr(res, "content") else str(res).strip()
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Error in generate node: {e}")
            return {"answer": f"Error generating answer: {str(e)}"}

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
