# Chroma

Chroma is a vector database commonly used in RAG demos. In a local setup, it can persist
its index to a folder on disk. Your raw source documents remain in a docs folder, while
the vector database stores chunked text, embeddings, and metadata for similarity search.

This makes Chroma a good first choice for small local projects because setup is simple and
you can inspect both the source files and the generated index.

## Why Chroma Fits a Local Tutorial

Chroma works well in a tutorial because it removes a lot of infrastructure concerns. You do
not need to provision a hosted service, configure network access, or design a database schema
before you can test retrieval. Instead, you can focus on the RAG pipeline itself: load files,
split them into chunks, turn those chunks into embeddings, store them, and retrieve them later
with a semantic query. That makes it easier to debug each stage in isolation.

## What Chroma Stores

When you add documents to Chroma, it stores more than just one big block of text. It stores
chunked text, the embedding vectors for those chunks, and metadata such as the source file name.
Later, when you ask a question, the retriever embeds the question, compares that vector against
stored vectors, and returns the closest matching chunks. In a tutorial project, this is useful
because you can inspect the returned chunks and see exactly which file they came from.

## How Chroma Helps Retrieval

The main job of Chroma in a RAG system is retrieval, not answer generation. Chroma does not
decide the final answer. Instead, it narrows the search space by returning the chunks that are
most relevant to the question. The language model then reads those chunks as context and writes
the final answer. This separation is important because it lets you debug retrieval quality before
you debug prompt design or answer generation quality.
