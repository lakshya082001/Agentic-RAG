import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def retrieve(store_type: str, user_query: str, vectorstore, embedder, k: int = 4, **kwargs):
    if store_type in ("chroma", "faiss"):
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 20},
        )
        return retriever.invoke(user_query)

    # raw chromadb collection path — uses LangChain MMR
    query_emb = embedder.embed_query(user_query)
    retrieved = vectorstore.query(
        query_embeddings=[query_emb],
        n_results=20,
        include=["documents", "metadatas", "embeddings"],
    )

    selected_indices = maximal_marginal_relevance(
        np.array(query_emb),
        np.array(retrieved["embeddings"][0]),
        lambda_mult=kwargs.get("lambda_mult", 0.5),
        k=k,
    )

    return [
        Document(
            page_content=retrieved["documents"][0][i],
            metadata=retrieved["metadatas"][0][i] or {},
        )
        for i in selected_indices
    ]


def hybrid_retrieve(store_type: str, user_query: str, vectorstore, chunks: list, k: int = 4):
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20},
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],
    )

    return ensemble_retriever.invoke(user_query)
