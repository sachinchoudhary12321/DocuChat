"""
Configuration management for DocuChat using Pydantic Settings.
Configures Groq LLM, Sentence-Transformers embeddings, FAISS vector store,
file upload limits, and memory storage paths.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_REPORTS_DIR = DATA_DIR / "sample_reports"
VECTOR_DIR = DATA_DIR / "vectorstore"
DB_PATH = DATA_DIR / "chat_memory.db"

# Ensure required runtime directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    # Application Metadata
    APP_NAME: str = "DocuChat - Multi-Document Conversational RAG"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # LLM Settings (Groq API)
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API Key for Llama 3.x models")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Groq model ID")
    GROQ_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=1.0)
    MAX_TOKENS: int = 1500

    # Embeddings Configuration
    # Sentence-Transformers running locally (Groq does not provide embeddings)
    EMBEDDING_PROVIDER: Literal["local", "sentence_transformers"] = "local"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking Parameters
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval Parameters
    TOP_K: int = 4
    RELEVANCE_THRESHOLD: float = 0.25

    # Storage Paths
    UPLOAD_PATH: Path = UPLOAD_DIR
    VECTOR_STORE_PATH: Path = VECTOR_DIR
    SQLITE_DB_PATH: Path = DB_PATH
    MAX_UPLOAD_SIZE_MB: int = 250

    # Conversational Memory
    MAX_HISTORY_TURNS: int = 8

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
