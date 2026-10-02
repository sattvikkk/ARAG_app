# Embeddings

Embeddings convert text into vectors. Similar meaning produces vectors that are close to
each other in vector space. This lets a retriever find semantically related chunks even
when the question uses different words than the original document.

In a local RAG project, a dedicated embedding model is usually separate from the chat
model. For example, you might use qwen for generation and nomic-embed-text for retrieval.
