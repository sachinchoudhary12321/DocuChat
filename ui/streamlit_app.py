"""
Streamlit Web Application for DocuChat.
Provides multi-PDF drag-and-drop upload, document browser,
conversational chat with LangGraph orchestration, query reformulation pills,
and page-level citation cards.

Architecture:
- 100% Session & Visitor Isolation: Each visitor starts with an empty workspace.
  Vector stores and chat histories are strictly session-scoped in st.session_state.
- Grounded Retrieval: Supports both document topic/overview queries and specific fact lookups.
- Page Citations: Exact document name and page number citations [Doc.pdf, p. X].
"""

import os
import sys
import uuid
import importlib
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure updated app submodules are freshly reloaded if running in long-lived container
for mod_name in ["app.config", "app.vectorstore", "app.memory", "app.graph", "app.ingestion", "app.prompts"]:
    if mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

import streamlit as st
from app.config import settings, UPLOAD_DIR
from app.ingestion import process_pdf_document
from app.vectorstore import FAISSVectorStoreManager, LocalDenseEmbeddings
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
        font-size: 1.02rem;
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


# ============================================================================
# MULTI-TENANT SESSION INITIALIZATION (ZERO-LEAK PRIVACY)
# ============================================================================

# Unique visitor session ID (generated once per browser tab / user session)
if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = f"usr_{uuid.uuid4().hex[:10]}"

# Ephemeral session-isolated vector store (never written to global disk)
if "vector_store" not in st.session_state:
    try:
        vs = FAISSVectorStoreManager(persist_dir=None, embeddings=LocalDenseEmbeddings())
    except TypeError:
        vs = FAISSVectorStoreManager(persist_dir=None)
    vs.session_id = st.session_state.user_session_id
    st.session_state.vector_store = vs

# Session-isolated memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

# LangGraph workflow tied to the user's isolated vector store
if "graph" not in st.session_state:
    st.session_state.graph = build_docuchat_graph(st.session_state.vector_store)

# Chat conversation ID within this user's session
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = f"{st.session_state.user_session_id}_chat_1"

# Chat message list for the UI
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Optional Groq API key configuration
if "groq_api_key" not in st.session_state:
    # Check Streamlit Cloud secrets or environment variable
    secret_key = None
    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    st.session_state.groq_api_key = secret_key or os.environ.get("GROQ_API_KEY", "")


vector_store: FAISSVectorStoreManager = st.session_state.vector_store
memory: ConversationMemory = st.session_state.memory
graph = st.session_state.graph

# Retrieve only the current visitor's indexed documents
docs = vector_store.get_indexed_documents()
total_chunks = sum(d["chunk_count"] for d in docs)


def refresh_history(chat_id: str):
    msgs = memory.get_messages(chat_id)
    st.session_state.chat_messages = [
        {"role": m["role"], "content": m["content"], "sources": m["sources"]}
        for m in msgs
    ]


# ============================================================================
# SIDEBAR: DOCUMENT INGESTION & CONTROLS
# ============================================================================
with st.sidebar:
    st.title("📑 DocuChat Controls")
    st.caption(f"Session: `{st.session_state.user_session_id}` • Private & Isolated")
    st.markdown("---")

    # Document Upload Section
    st.subheader("📤 Upload Your PDF")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload any PDF (technical documentation, reports, papers, manuals)"
    )

    if uploaded_files:
        if st.button("🚀 Process & Index PDF(s)", type="primary", use_container_width=True):
            success_count = 0
            with st.spinner("Extracting text, chunking & indexing in private session store..."):
                user_upload_dir = UPLOAD_DIR / st.session_state.user_session_id
                user_upload_dir.mkdir(parents=True, exist_ok=True)

                for file in uploaded_files:
                    save_path = user_upload_dir / file.name
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())

                    try:
                        chunks, stats = process_pdf_document(save_path)
                        vector_store.add_documents(chunks)
                        st.sidebar.success(f"✓ **{file.name}** indexed ({stats['total_chunks']} chunks, {stats['total_pages']} pages)")
                        success_count += 1
                    except Exception as e:
                        st.sidebar.error(f"✗ **{file.name}**: {e}")
                    finally:
                        # Clean up temporary uploaded file on disk
                        try:
                            if save_path.exists():
                                save_path.unlink()
                        except Exception:
                            pass

            if success_count > 0:
                st.rerun()

    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    st.metric("Total Indexed Documents", len(docs), f"{total_chunks} Chunks")

    if docs:
        for d in docs:
            with st.expander(f"📄 {d['filename']}", expanded=False):
                st.write(f"• **Pages:** {d['total_pages']}")
                st.write(f"• **Chunks:** {d['chunk_count']}")
                st.write(f"• **Indexed Pages:** {', '.join(map(str, d['pages_indexed']))}")
                if st.button("🗑️ Delete", key=f"del_{d['filename']}", use_container_width=True):
                    vector_store.delete_document(d['filename'])
                    st.warning(f"Deleted {d['filename']}")
                    st.rerun()
    else:
        st.info("No documents currently indexed in your session. Upload a PDF above to get started!")

    st.markdown("---")
    st.subheader("💬 Session Manager")
    
    # List only chats belonging to this visitor's session
    visitor_sessions = memory.list_sessions(prefix=st.session_state.user_session_id)
    chat_ids = [s["session_id"] for s in visitor_sessions]
    if st.session_state.current_chat_id not in chat_ids:
        chat_ids.insert(0, st.session_state.current_chat_id)

    # Format human-friendly labels (e.g. Chat 1, Chat 2)
    def format_chat_label(cid: str) -> str:
        parts = cid.split("_chat_")
        if len(parts) == 2:
            return f"Chat #{parts[1]}"
        return cid

    selected_chat = st.selectbox(
        "Active Conversation",
        chat_ids,
        index=chat_ids.index(st.session_state.current_chat_id),
        format_func=format_chat_label
    )
    if selected_chat != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat
        refresh_history(selected_chat)
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New Chat", use_container_width=True):
            new_chat_num = len(chat_ids) + 1
            new_chat_id = f"{st.session_state.user_session_id}_chat_{new_chat_num}"
            st.session_state.current_chat_id = new_chat_id
            st.session_state.chat_messages = []
            st.rerun()
    with c2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            memory.clear_session(st.session_state.current_chat_id)
            st.session_state.chat_messages = []
            st.rerun()

    # Document Filtering
    doc_filter_options = ["All Documents"] + [d["filename"] for d in docs]
    selected_filter = st.selectbox("Filter Search Scope", doc_filter_options)
    active_filter = None if selected_filter == "All Documents" else selected_filter

    # Advanced Settings Expander
    with st.expander("⚙️ Advanced Settings", expanded=False):
        api_input = st.text_input(
            "Groq API Key (Optional)",
            value=st.session_state.groq_api_key,
            type="password",
            help="Leave blank to use the high-speed local deterministic RAG engine."
        )
        if api_input != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_input
            if api_input:
                os.environ["GROQ_API_KEY"] = api_input
                settings.GROQ_API_KEY = api_input
            else:
                os.environ.pop("GROQ_API_KEY", None)
                settings.GROQ_API_KEY = None
            st.session_state.graph = build_docuchat_graph(st.session_state.vector_store)
            st.rerun()


# ============================================================================
# MAIN INTERFACE: CHAT & CITATIONS
# ============================================================================
st.markdown('<div class="header-title">📑 DocuChat: Multi-Document Q&A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-sub">'
    'Upload your own PDFs in the sidebar and ask questions in natural language. '
    'Answers are strictly grounded in your documents with exact page citations.'
    '</div>',
    unsafe_allow_html=True
)

if docs:
    active_names = ", ".join([d["filename"] for d in docs])
    st.success(f"📚 **Active Knowledge Base:** {active_names} ({total_chunks} indexed chunks ready)")

    # Dynamic suggestions for the user's uploaded documents
    st.markdown("**💡 Quick Question Suggestions:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📌 Summarize Document", use_container_width=True):
            st.session_state.prompt_input = "Tell me the topic and a summary of the uploaded file."
    with col2:
        if st.button("🔍 Key Concepts", use_container_width=True):
            st.session_state.prompt_input = "What are the main key concepts or definitions explained?"
    with col3:
        if st.button("📊 Key Data & Facts", use_container_width=True):
            st.session_state.prompt_input = "What are the key statistics, facts, or numbers provided?"
    with col4:
        if st.button("🚫 Out-of-Scope Test", use_container_width=True):
            st.session_state.prompt_input = "What is the policy on employee dental surgery reimbursement?"
else:
    st.info(
        "👈 **Your workspace is currently empty.**\n\n"
        "To get started:\n"
        "1. Upload one or more PDF files using the **'Upload Your PDF'** tool in the left sidebar.\n"
        "2. Click **'🚀 Process & Index PDF(s)'** to extract and index text.\n"
        "3. Ask questions in natural language and receive strictly grounded answers with exact page citations!\n\n"
        "*Note: Your documents and chat history are 100% private and isolated to your browser session.*"
    )


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
    if not docs:
        st.warning("⚠️ Please upload and index at least one PDF in the left sidebar first before asking questions.")
    else:
        st.session_state.chat_messages.append({"role": "user", "content": user_query, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document context & running LangGraph flow..."):
                chat_id = st.session_state.current_chat_id
                history = memory.get_langchain_messages(chat_id, limit=settings.MAX_HISTORY_TURNS)

                state_input = {
                    "question": user_query,
                    "session_id": chat_id,
                    "chat_history": history,
                    "standalone_question": user_query,
                    "documents": [],
                    "sources": [],
                    "is_relevant": False,
                    "answer": "",
                    "doc_filter": active_filter
                }

                final_state = graph.invoke(state_input)

                # Record in isolated memory
                memory.add_message(chat_id, "user", user_query)
                memory.add_message(chat_id, "assistant", final_state["answer"], sources=final_state["sources"])

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
