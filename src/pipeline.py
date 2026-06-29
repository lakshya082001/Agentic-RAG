import os
from langchain_core.tools import tool
from langchain_openai.embeddings import OpenAIEmbeddings

from src.ingestion import read_sources
from src.chunking import chunked
from src.vectorstore import create_vectorstore
from src.retrieval import retrieve, hybrid_retrieve
from src.reranker import rerank

URLS = [
    "https://www.cms.gov/medicare-coverage-database/view/ncacal-decision-memo.aspx?proposed=N&NCAId=204"
]
PDFS_PATH = "./Data"


@tool(description="""Search and retrieve relevant documents.
store_type options:
- 'faiss': fast, URLs or raw text only
- 'chroma': PDFs with metadata filtering
- 'chroma_persistent': multi-source, persists across calls
Auto-selected as faiss if not provided.""")
def rag_pipeline(user_query: str, store_type: str = "faiss") -> list:
    print(f"\n[RAG] Tool called | store_type='{store_type}' | query='{user_query}'")

    embedder = OpenAIEmbeddings(
        base_url=os.environ.get("OPENAI_API_BASE"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
    )

    print("[RAG] 1/4 Ingesting sources...")
    docs = read_sources(URLS, PDFS_PATH, raw_texts=[])
    print(f"[RAG] 1/4 Done — {len(docs)} documents loaded")

    print("[RAG] 2/4 Chunking...")
    chunked_docs = chunked(docs, chunking_style="recursive")
    print(f"[RAG] 2/4 Done — {len(chunked_docs)} chunks")

    print("[RAG] 3/4 Building vectorstore & retrieving...")
    if store_type not in ("faiss", "chroma"):
        vectorstore, texts, text_embeddings = create_vectorstore(store_type, chunked_docs, embedder)
        retrieved_results = retrieve(
            store_type, user_query, vectorstore, embedder,
            texts=texts, embeddings=text_embeddings, k=20,
        )
    else:
        vectorstore = create_vectorstore(store_type, chunked_docs, embedder)
        retrieved_results = hybrid_retrieve(store_type, user_query, vectorstore, chunked_docs, k=20)
    print(f"[RAG] 3/4 Done — {len(retrieved_results)} results retrieved")

    print("[RAG] 4/4 Reranking...")
    final = rerank(retrieved_results, user_query, k=4)
    print(f"[RAG] 4/4 Done — {len(final)} results after rerank\n")

    return final
