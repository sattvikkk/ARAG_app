import time
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "langchain_native_rag"
EMBEDDING_MODEL = "embeddinggemma"
CHAT_MODEL = "qwen3:4b"


def retrieval(question: str) -> list[Document]:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} documents for the question: {question}")
    return docs


def generate_response(question: str, docs: list[Document]) -> tuple[AIMessage, list[str]]:
    start = time.time()
    prompt = ChatPromptTemplate.from_template(
        """
you are answering questions based on the following context." 
context:{context}
question:{question}
instructions: Please provide a concise and accurate answer based on the context provided. If the answer is not present in the context, respond with "I don't know."
"""
    )
    llm = ChatOllama(model=CHAT_MODEL)
    chain = prompt | llm
    context = "\n\n".join([
        f"Source: {doc.metadata['source']} \nContent: {doc.page_content}" for doc in docs]
    )
    sources = sorted(set(doc.metadata['source'] for doc in docs))

    response = chain.invoke({
        "context": context,
        "question": question
    })

    end = time.time()
    print(f"Response generated in {end - start:.2f} seconds.")
    return response, sources


def main() -> None:
    question = "What is Chroma and why would I use it in a local RAG project?"
    docs = retrieval(question)
    if not docs:
        print("No relevant documents found for the question.")
        return
    response, sources = generate_response(question, docs)
    print(f"Answer: {response.content}")
    print(f"Sources: {f'Sources: {sources}'}")


if __name__ == "__main__":
    main()
