from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder


def rerank(docs: list, user_query: str, k: int = 4) -> list:
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    reranker = CrossEncoderReranker(model=model, top_n=k)
    return reranker.compress_documents(docs, user_query)
