import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def path(env_var: str, default: str) -> str:
    """Get the path from an environment variable or return the default."""
    raw = os.getenv(env_var, default)
    path = Path(raw)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


@dataclass
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
    eval_model: str = os.getenv(
        "EVAL_MODEL", os.getenv("LLM_MODEL", "qwen2.5:3b"))
    sql_model: str = os.getenv(
        "SQL_MODEL", os.getenv("LLM_MODEL", "qwen2.5:3b"))
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", 0))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    chroma_dir: str = field(
        default_factory=lambda: path("CHROMA_DIR", "chroma"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")
    document_dir: str = field(default_factory=lambda: path(
        "DOCUMENT_DIR", "data/document"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))

    sqlite_path: str = field(default_factory=lambda: path(
        "SQLITE_PATH", "data/sample.db"))
    tables_dir: str = field(
        default_factory=lambda: path("TABLES_DIR", "data/table"))

    kb_description: str = os.getenv("KB_DESCRIPTION",
                                    "Documentation about Retrieval-Augmented Generation (RAG) and related concepts."
                                    "pattern with langchain, LLMs, and vector databases. This knowledge base is designed to provide insights into the architecture, implementation, and best practices for building RAG systems.")

    sql_description: str = os.getenv("SQL_DESCRIPTION",
                                     "Documentation about SQL and related concepts. This knowledge base is designed to provide insights into the"
                                     "architecture, implementation, and best practices for building SQL systems.")


settings = Settings()
