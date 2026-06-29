import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from chromadb import PersistentClient


def create_vectorstore(store_type: str, chunked_docs: list[Document], embedder):
    if store_type == "chroma":
        return Chroma.from_documents(
            chunked_docs,
            embedding=embedder,
            collection_name="rag_docs_chroma",
            persist_directory="./db",
        )

    if store_type == "faiss":
        sample_embedding = embedder.embed_query("test")
        dimension = len(sample_embedding)

        index_hnsw = faiss.IndexHNSWFlat(dimension, 32)
        index_hnsw.hnsw.efConstruction = 200
        index_hnsw.hnsw.efSearch = 50

        store = FAISS.from_documents(chunked_docs, embedding=embedder)
        store.index = index_hnsw
        return store

    # chromadb raw collection (default / custom)
    texts = [doc.page_content for doc in chunked_docs]
    embeddings = embedder.embed_documents(texts)

    client = PersistentClient(path="./db")
    collection = client.get_or_create_collection("rag_docs")
    collection.add(
        ids=[str(i) for i in range(len(texts))],
        documents=texts,
        embeddings=embeddings,
    )
    return collection, texts, embeddings
