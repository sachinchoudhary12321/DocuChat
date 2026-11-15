"""
Streamlit Web Application for DocuChat.
Provides multi-PDF drag-and-drop upload, document browser,
conversational chat with LangGraph orchestration, query reformulation pills,
and page-level citation cards.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app.config import settings, UPLOAD_DIR, SAMPLE_REPORTS_DIR
from app.ingestion import process_pdf_document
from app.vectorstore import FAISSVectorStoreManager
from app.memory import ConversationMemory
from app.graph import build_docuchat_graph

st.set_page_config(
    page_title="DocuChat | Multi-Document RAG",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .header-sub {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 6px;
        font-size: 0.88rem;
    }
    .pill {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 12px;
        background-color: #E2E8F0;
        color: #1E293B;
        margin-right: 6px;
    }
    .pill-score {
        background-color: #DCFCE7;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_system():
    vs = FAISSVectorStoreManager()
    mem = ConversationMemory()
    workflow = build_docuchat_graph(vs)
    return vs, mem, workflow


vector_store, memory, graph = init_system()

# Load currently indexed documents
indexed_docs = vector_store.get_indexed_documents()

# Session State Management
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "Session-1"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


def refresh_history(session_id: str):
    msgs = memory.get_messages(session_id)
    st.session_state.chat_messages = [
        {"role": m["role"], "content": m["content"], "sources": m["sources"]}
        for m in msgs
    ]


if not st.session_state.chat_messages:
    refresh_history(st.session_state.current_session_id)


# ============================================================================
# SIDEBAR: DOCUMENT INGESTION & CONTROLS
# ============================================================================
with st.sidebar:
    st.title("📑 DocuChat Controls")
    st.caption("Multi-Document Conversational RAG • LangGraph & Groq")
    st.markdown("---")

    # Document Upload Section
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF reports",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple enterprise reports (financial, policy, runbooks)"
    )

    if uploaded_files:
        if st.button("⚡ Index Uploaded PDFs", use_container_width=True):
            with st.spinner("Extracting text, chunking & creating vector embeddings..."):
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                for file in uploaded_files:
                    save_path = UPLOAD_DIR / file.name
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())

                    try:
                        chunks, stats = process_pdf_document(save_path)
                        vector_store.add_documents(chunks)
                        st.success(f"✓ **{file.name}** ({stats['total_chunks']} chunks, {stats['total_pages']} pages)")
                    except Exception as e:
                        st.error(f"✗ Failed {file.name}: {e}")
            st.rerun()

    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    docs = vector_store.get_indexed_documents()
    total_chunks = sum(d["chunk_count"] for d in docs)
    st.metric("Total Indexed Documents", len(docs), f"{total_chunks} Chunks")

    if docs:
        for d in docs:
            with st.expander(f"📄 {d['filename']}", expanded=False):
                st.write(f"• **Doc ID:** `{d['doc_id']}`")
                st.write(f"• **Pages:** {d['total_pages']}")
                st.write(f"• **Chunks:** {d['chunk_count']}")
                st.write(f"• **Indexed Pages:** {', '.join(map(str, d['pages_indexed']))}")
                if st.button("🗑️ Delete", key=f"del_{d['filename']}", use_container_width=True):
                    vector_store.delete_document(d['filename'])
                    st.warning(f"Deleted {d['filename']}")
                    st.rerun()
    else:
        st.info("No documents currently indexed.")

    st.markdown("---")
    st.subheader("💬 Session Manager")
    sessions = memory.list_sessions()
    sess_ids = [s["session_id"] for s in sessions]
    if st.session_state.current_session_id not in sess_ids:
        sess_ids.append(st.session_state.current_session_id)

    selected_sess = st.selectbox("Active Session", sess_ids, index=sess_ids.index(st.session_state.current_session_id))
    if selected_sess != st.session_state.current_session_id:
        st.session_state.current_session_id = selected_sess
        refresh_history(selected_sess)
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New Chat", use_container_width=True):
            new_sess = f"Session-{len(sess_ids) + 1}"
            st.session_state.current_session_id = new_sess
            st.session_state.chat_messages = []
            st.rerun()
    with c2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            memory.clear_session(st.session_state.current_session_id)
            st.session_state.chat_messages = []
            st.rerun()

    # Document Filtering
    doc_filter_options = ["All Documents"] + [d["filename"] for d in docs]
    selected_filter = st.selectbox("Filter Search Scope", doc_filter_options)
    active_filter = None if selected_filter == "All Documents" else selected_filter

    if st.button("🔄 Reset to Default Reports", use_container_width=True):
        vector_store.clear()
        if SAMPLE_REPORTS_DIR.exists():
            for sf in SAMPLE_REPORTS_DIR.glob("*.pdf"):
                c, _ = process_pdf_document(sf)
                vector_store.add_documents(c)
        st.rerun()


# ============================================================================
# MAIN INTERFACE: CHAT & CITATIONS
# ============================================================================
st.markdown('<div class="header-title">📑 DocuChat: Intelligent Multi-Document RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-sub">'
    'Ask questions across your multi-document collection. Answers are grounded in document context '
    'with exact page citations and multi-turn conversational memory.'
    '</div>',
    unsafe_allow_html=True
)

# Example Query Chips
st.markdown("**💡 Quick Test Prompts:**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📊 TechCorp Q3 Revenue", use_container_width=True):
        st.session_state.prompt_input = "What was TechCorp total revenue in Q3 2024 and year-over-year growth?"
with col2:
    if st.button("🚨 P1 Incident Response SLA", use_container_width=True):
        st.session_state.prompt_input = "What is the SLA response time and escalation lead for a P1 critical incident?"
with col3:
    if st.button("🛡️ GDPR SAR Request SLA", use_container_width=True):
        st.session_state.prompt_input = "What is the mandatory resolution window for Subject Access Requests (SAR) under GDPR?"
with col4:
    if st.button("🚫 Out-of-Scope Fallback", use_container_width=True):
        st.session_state.prompt_input = "What is the employee health insurance reimbursement limit for dental surgery?"


# Render Conversation Messages
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} Supporting Sources", expanded=False):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"""
                    <div class="citation-card">
                        <div>
                            <span class="pill">Citation {idx+1}</span>
                            <b>📄 {src['filename']}</b> (p. {src['page']})
                            <span class="pill pill-score">Score: {src.get('score', 'N/A')}</span>
                        </div>
                        <p style="margin-top:6px; color:#475569; font-style: italic;">
                            "{src['snippet']}"
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

# User Chat Input
user_query = st.chat_input("Ask a question about your uploaded documents...")
if "prompt_input" in st.session_state and st.session_state.prompt_input:
    user_query = st.session_state.prompt_input
    st.session_state.prompt_input = None

if user_query:
    st.session_state.chat_messages.append({"role": "user", "content": user_query, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Rewriting query, searching FAISS index & running LangGraph flow..."):
            session_id = st.session_state.current_session_id
            history = memory.get_langchain_messages(session_id, limit=settings.MAX_HISTORY_TURNS)

            state_input = {
                "question": user_query,
                "session_id": session_id,
                "chat_history": history,
                "standalone_question": user_query,
                "documents": [],
                "sources": [],
                "is_relevant": False,
                "answer": "",
                "doc_filter": active_filter
            }

            final_state = graph.invoke(state_input)

            # Record in SQLite memory
            memory.add_message(session_id, "user", user_query)
            memory.add_message(session_id, "assistant", final_state["answer"], sources=final_state["sources"])

            # Render Query Reformulation Pill if rewritten
            if final_state["standalone_question"].strip() != user_query.strip():
                st.info(f"🔄 **Query Reformulation:** *\"{final_state['standalone_question']}\"*")

            st.markdown(final_state["answer"])

            # Render Supporting Source Cards
            if final_state["sources"]:
                with st.expander(f"📚 {len(final_state['sources'])} Supporting Sources", expanded=True):
                    for idx, src in enumerate(final_state["sources"]):
                        st.markdown(f"""
                        <div class="citation-card">
                            <div>
                                <span class="pill">Citation {idx+1}</span>
                                <b>📄 {src['filename']}</b> (p. {src['page']})
                                <span class="pill pill-score">Relevance: {src.get('score', 'N/A')}</span>
                            </div>
                            <p style="margin-top:6px; color:#475569; font-style: italic;">
                                "{src['snippet']}"
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": final_state["answer"],
        "sources": final_state["sources"]
    })
