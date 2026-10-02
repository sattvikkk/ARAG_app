from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = ROOT_DIR / "source_data"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "langchain_native_rag"
EMBEDDING_MODEL = "embeddinggemma"


def load_markdown_documents() -> list[Document]:
    """Load markdown documents from the source directory."""
    documents: list[Document] = []
    for path in sorted(SOURCE_DIR.glob("**/*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "doc_type": "markdown",
                },
            )
        )

    print(f"Loaded {len(documents)} markdown file(s) from {SOURCE_DIR}")
    return documents


def embedding(documents: list[Document]) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    vectorstore.reset_collection()

    vectorstore.add_documents(chunks)

    print(f"Added {len(chunks)} documents to the vector store at {CHROMA_DIR}")


def main() -> None:
    documents = load_markdown_documents()
    if not documents:
        print("No markdown documents found in the source directory.")
        return
    embedding(documents)


if __name__ == "__main__":
    main()
