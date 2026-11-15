"""
Conversational memory management for DocuChat.
Persists multi-session dialogue history in SQLite with sliding window support
and LangChain message object conversion.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.config import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Session-isolated conversation store backed by SQLite.
    Stores user queries, assistant responses, and structured citations.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or settings.SQLITE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_turns(session_id);
            """)
            conn.commit()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Add a single message turn to the session history."""
        sources_str = json.dumps(sources) if sources else None
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversation_turns (session_id, role, content, sources_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, sources_str)
            )
            conn.commit()
            return cursor.lastrowid

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve recent turns for session ordered chronologically."""
        max_records = limit or (settings.MAX_HISTORY_TURNS * 2)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, role, content, sources_json, timestamp
                FROM conversation_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, max_records)
            )
            rows = cursor.fetchall()

        results = []
        for r in reversed(rows):
            sources = json.loads(r["sources_json"]) if r["sources_json"] else []
            results.append({
                "id": r["id"],
                "session_id": r["session_id"],
                "role": r["role"],
                "content": r["content"],
                "sources": sources,
                "timestamp": str(r["timestamp"])
            })
        return results

    def get_langchain_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """Convert stored session turns into LangChain BaseMessage objects."""
        raw_msgs = self.get_messages(session_id, limit=limit)
        lc_msgs: List[BaseMessage] = []
        for m in raw_msgs:
            if m["role"] == "user":
                lc_msgs.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_msgs.append(AIMessage(content=m["content"]))
        return lc_msgs

    def clear_session(self, session_id: str) -> int:
        """Erase dialogue memory for a given session ID."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with message counts and timestamps."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id,
                       COUNT(*) as message_count,
                       MAX(timestamp) as last_active
                FROM conversation_turns
                GROUP BY session_id
                ORDER BY last_active DESC
            """)
            rows = cursor.fetchall()
        return [dict(r) for r in rows]
