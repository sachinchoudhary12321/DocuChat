"""
FastAPI REST API for DocuChat.
Provides endpoints for multi-PDF uploads, document management,
conversational RAG querying, session history retrieval, and health checks.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.config import settings, UPLOAD_DIR, SAMPLE_REPORTS_DIR
from app.ingestion import process_pdf_document, validate_pdf_file
from app.vectorstore import FAISSVectorStoreManager
from app.memory import ConversationMemory
from app.graph import build_docuchat_graph

logger = logging.getLogger("docuchat_api")

# Singletons for FastAPI runtime
vector_store = FAISSVectorStoreManager()
memory = ConversationMemory()
graph = build_docuchat_graph(vector_store)


# ============================================================================
# PYDANTIC REQUEST & RESPONSE SCHEMAS
# ============================================================================

class SourceItem(BaseModel):
    filename: str = Field(..., description="Source document file name")
    page: int = Field(..., description="1-indexed page number")
    score: float = Field(..., description="Relevance / similarity score")
    chunk_id: str = Field(..., description="Unique chunk ID")
    snippet: str = Field(..., description="Text excerpt from the source chunk")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question or follow-up")
    session_id: Optional[str] = Field(default="default_session", description="Session identifier for conversational memory")
    doc_filter: Optional[str] = Field(default=None, description="Optional document name or doc_id filter")


class ChatResponse(BaseModel):
    session_id: str
    question: str
    standalone_question: str
    answer: str
    sources: List[SourceItem]
    model_used: str


class DocumentItem(BaseModel):
    doc_id: str
    filename: str
    total_pages: int
    chunk_count: int
    pages_indexed: List[int]


class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentItem]


class UploadResultItem(BaseModel):
    status: str
    filename: str
    doc_id: Optional[str] = None
    total_pages: int
    total_chunks: int
    message: str


class HistoryItem(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    sources: List[Dict[str, Any]]
    timestamp: str


class HistoryResponse(BaseModel):
    session_id: str
    total_turns: int
    messages: List[HistoryItem]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    total_documents_indexed: int
    total_chunks_indexed: int
    llm_provider: str
    embedding_model: str


# ============================================================================
# LIFESPAN & APPLICATION SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    yield


app = FastAPI(
    title="DocuChat API",
    description="Multi-Document RAG with Conversational Memory, Groq Llama 3, and LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def get_health():
    """System health check and index diagnostics."""
    docs = vector_store.get_indexed_documents()
    total_chunks = sum(d["chunk_count"] for d in docs)
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        total_documents_indexed=len(docs),
        total_chunks_indexed=total_chunks,
        llm_provider="Groq (" + settings.GROQ_MODEL + ")" if settings.GROQ_API_KEY else "Local-Deterministic",
        embedding_model=settings.EMBEDDING_MODEL_NAME
    )


@app.post("/documents/upload", response_model=List[UploadResultItem], tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload one or more PDF documents, parse, split into chunks, and index in FAISS."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    results = []
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append(UploadResultItem(
                status="error",
                filename=file.filename,
                total_pages=0,
                total_chunks=0,
                message="Only PDF files are supported."
            ))
            continue

        save_path = UPLOAD_DIR / file.filename
        try:
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            chunks, stats = process_pdf_document(save_path)
            vector_store.add_documents(chunks)

            results.append(UploadResultItem(
                status="success",
                filename=file.filename,
                doc_id=stats["doc_id"],
                total_pages=stats["total_pages"],
                total_chunks=stats["total_chunks"],
                message=f"Indexed {stats['total_chunks']} chunks ({stats['total_pages']} pages)."
            ))
        except Exception as e:
            results.append(UploadResultItem(
                status="error",
                filename=file.filename,
                total_pages=0,
                total_chunks=0,
                message=str(e)
            ))

    return results


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all indexed documents with page numbers and chunk counts."""
    docs = vector_store.get_indexed_documents()
    items = [
        DocumentItem(
            doc_id=d["doc_id"],
            filename=d["filename"],
            total_pages=d["total_pages"],
            chunk_count=d["chunk_count"],
            pages_indexed=d["pages_indexed"]
        )
        for d in docs
    ]
    return DocumentListResponse(
        total_documents=len(items),
        total_chunks=sum(d["chunk_count"] for d in docs),
        documents=items
    )


@app.delete("/documents/{doc_identifier}", tags=["Documents"])
async def delete_document(doc_identifier: str):
    """Remove a document from the vector store by filename or doc_id."""
    removed = vector_store.delete_document(doc_identifier)
    if removed == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_identifier}' not found in index."
        )

    # Clean upload file if present
    target = UPLOAD_DIR / doc_identifier
    if target.exists():
        try:
            target.unlink()
        except Exception:
            pass

    return {
        "status": "success",
        "identifier": doc_identifier,
        "chunks_removed": removed,
        "message": f"Successfully deleted '{doc_identifier}' ({removed} chunks removed)."
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Execute conversational RAG query through LangGraph:
    1. Rewrites follow-up questions using session memory.
    2. Retrieves top-k chunks from FAISS vector store.
    3. Evaluates relevance with Relevance Grader.
    4. Synthesizes grounded answer with inline citations or falls back safely.
    5. Records dialogue turn in SQLite memory.
    """
    if not request.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    session_id = request.session_id or "default_session"
    chat_history = memory.get_langchain_messages(session_id, limit=settings.MAX_HISTORY_TURNS)

    initial_state = {
        "question": request.question,
        "session_id": session_id,
        "chat_history": chat_history,
        "standalone_question": request.question,
        "documents": [],
        "sources": [],
        "is_relevant": False,
        "answer": "",
        "doc_filter": request.doc_filter
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"Error invoking LangGraph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph execution failed: {str(e)}"
        )

    # Persist interaction into session memory
    memory.add_message(session_id, "user", request.question)
    memory.add_message(session_id, "assistant", final_state["answer"], sources=final_state["sources"])

    source_items = [
        SourceItem(
            filename=s["filename"],
            page=s["page"],
            score=s["score"],
            chunk_id=s["chunk_id"],
            snippet=s["snippet"]
        )
        for s in final_state["sources"]
    ]

    return ChatResponse(
        session_id=session_id,
        question=request.question,
        standalone_question=final_state["standalone_question"],
        answer=final_state["answer"],
        sources=source_items,
        model_used=settings.GROQ_MODEL if settings.GROQ_API_KEY else "local-deterministic"
    )


@app.get("/chat/{session_id}/history", response_model=HistoryResponse, tags=["Chat"])
async def get_chat_history(session_id: str):
    """Retrieve chronological message history for a given session."""
    messages = memory.get_messages(session_id)
    items = [
        HistoryItem(
            id=m["id"],
            session_id=m["session_id"],
            role=m["role"],
            content=m["content"],
            sources=m["sources"],
            timestamp=m["timestamp"]
        )
        for m in messages
    ]
    return HistoryResponse(
        session_id=session_id,
        total_turns=len(items),
        messages=items
    )


@app.delete("/chat/{session_id}", tags=["Chat"])
async def clear_chat_history(session_id: str):
    """Clear dialogue history for a session."""
    count = memory.clear_session(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "messages_deleted": count,
        "message": f"Cleared memory for session '{session_id}' ({count} records deleted)."
    }
