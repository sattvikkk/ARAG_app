import logging
from typing import List, Optional
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from arag_app.config import settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TEXT_SUFFIXES = [".txt", ".md", ".rst",
                 ".csv", ".json", ".jsonl", ".yaml", ".yml"]


@lru_cache(maxsize=1)
def get_embedding() -> HuggingFaceEmbeddings:
    """
    Get the HuggingFaceEmbeddings instance.

    Returns:
        HuggingFaceEmbeddings: The embeddings instance.
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model
                                 )


def load_documents(directory: str | None = None) -> list[Document]:
    base_path = Path(directory or settings.document_dir)
    documents: list[Document] = []
    for path in sorted(base_path.rglob("*")):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            documents.append(Document(page_content=path.read_text(encoding="utf-8"),
                                      metadata={"source": path.name}))
        elif path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            documents.extend(PyPDFLoader(str(path)).load())
    logger.info(f"Loaded {len(documents)} documents from {base_path}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                              chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)


def get_vectorstore(documents: list[Document], persist_directory: Optional[str] = None) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_dir,
        embedding_function=get_embedding(),
    )


def ingest(directory: str | None = None) -> int:
    chunks = split_documents(load_documents(directory))
    store = get_vectorstore(chunks)
    store.reset_collection()
    if chunks:
        store.add_documents(chunks)
    logger.info(f"Ingested {len(chunks)} chunks into the vector store")
    return len(chunks)


def ensure_index() -> None:
    if get_vectorstore([])._collection.count() == 0:
        logger.info("Vector store is empty, starting ingestion...")
        ingest()


def get_retriever(k: int | None = None):

    return get_vectorstore([]).as_retriever(search_kwargs={"k": k or settings.retrieval_k}
                                            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = ingest()
    print(f"Ingested {count} chunks into the vector store.")


if __name__ == "__main__":
    main()
