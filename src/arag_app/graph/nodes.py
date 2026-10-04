import logging
from typing import Any

from langchain_core.documents import Document

from ..config import settings
from ..ingestion import get_retriever
from ..tools.sql import run_sql_question
from ..tools.web_search import search_web
from .chains import (
    get_document_grader,
    get_generator,
    get_grounding_grader,
    get_rewriter,
    get_router,
    get_sql_llm,
)
from .state import GraphState

logger = logging.getLogger(__name__)


def _require_state(state: GraphState, key: str) -> Any:
    value = state.get(key)
    if value is None:
        raise KeyError(f"GraphState missing required key: {key}")
    return value


def _format_docs(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in documents
    )


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def route(state: GraphState) -> GraphState:
    """Decide which data source should handle the question."""
    question = _require_state(state, "question")
    result = get_router().invoke({"question": question})
    datasource = getattr(result, "datasource", None)
    if datasource is None and isinstance(result, dict):
        datasource = result.get("datasource")
    if datasource is None:
        raise ValueError("Router result did not include a datasource")
    logger.info("Router -> %s", datasource)
    return {
        "datasource": datasource,
        "original_question": question,
        "retries": 0,
    }


def retrieve(state: GraphState) -> GraphState:
    question = _require_state(state, "question")
    documents = get_retriever().invoke(question)
    logger.info("Retrieved %d chunks", len(documents))
    return {"documents": documents}


def grade_documents(state: GraphState) -> GraphState:
    """Keep only the retrieved chunks an LLM grader judges relevant."""
    question = _require_state(state, "question")
    documents = state.get("documents") or []
    grader = get_document_grader()
    relevant = [
        doc
        for doc in documents
        if grader.invoke({"document": doc.page_content, "question": question}) == "yes"
    ]
    logger.info("Document grading: %d/%d relevant",
                len(relevant), len(documents))
    return {"documents": relevant}


def rewrite_query(state: GraphState) -> GraphState:
    question = _require_state(state, "question")
    original_question = state.get("original_question") or question
    rewritten = get_rewriter().invoke(
        {
            "original_question": original_question,
            "question": question,
        }
    ).strip()
    logger.info("Rewrote query -> %s", rewritten)
    return {"question": rewritten, "retries": state.get("retries", 0) + 1}


def web_search(state: GraphState) -> GraphState:
    question = _require_state(state, "question")
    documents = search_web(question)
    logger.info("Web search returned %d results", len(documents))
    # Mark the source so downstream decisions know we are on the web path
    # (this node also serves as fallback when vectorstore retrieval fails).
    return {"documents": documents, "datasource": "web_search"}


def query_sql(state: GraphState) -> GraphState:
    question = _require_state(state, "question")
    document = run_sql_question(question, get_sql_llm())
    return {"documents": [document]}


SOURCE_NOTES = {
    "vectorstore": "the user's own indexed documents",
    "web_search": "a public web search — NOT the user's documents",
    "sql": "a SQL query over the user's database",
}


def generate(state: GraphState) -> GraphState:
    documents = state.get("documents") or []
    question = state.get("original_question") or _require_state(
        state, "question")
    datasource = state.get("datasource") or "unknown"
    generation = get_generator().invoke(
        {
            "context": _format_docs(documents),
            "question": question,
            "source_note": SOURCE_NOTES.get(datasource, "an unknown source"),
        }
    )
    return {"generation": generation}


# --------------------------------------------------------------------------
# Routing decisions (pure functions over the state)
# --------------------------------------------------------------------------
def select_datasource(state: GraphState) -> str:
    return str(state.get("datasource") or "")


def decide_after_grading(state: GraphState) -> str:
    """After grading: answer, retry with a better query, or fall back to web."""
    documents = state.get("documents") or []
    if documents:
        return "generate"
    if state.get("retries", 0) < settings.max_retries:
        return "rewrite"
    logger.info(
        "No relevant documents after %d retries, falling back to web", settings.max_retries)
    return "web_search"


def route_after_rewrite(state: GraphState) -> str:
    """Send the rewritten query back to the source that produced bad results."""
    return "web_search" if state.get("datasource") == "web_search" else "retrieve"


def grade_generation(state: GraphState) -> str:
    """Check the answer is grounded in the evidence; retry once if not.

    SQL results are deterministic query output, so grounding them against an
    LLM grader adds latency without value — accept them directly.
    """
    datasource = state.get("datasource")
    documents = state.get("documents") or []
    if datasource == "sql" or not documents:
        return "useful"
    generation = state.get("generation")
    if generation is None:
        return "useful"
    grounded = get_grounding_grader().invoke(
        {
            "documents": _format_docs(documents),
            "generation": generation,
        }
    )
    if grounded == "yes":
        return "useful"
    if state.get("retries", 0) < settings.max_retries:
        logger.info("Answer not grounded, rewriting query and retrying")
        return "rewrite"
    logger.info(
        "Answer not grounded but retry budget exhausted, returning anyway")
    return "useful"
