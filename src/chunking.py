from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


def chunked(
    docs: list[Document],
    chunking_style: str = "recursive",
    chunk_size: int = 100,
    chunk_overlap: int = 50,
) -> list[Document]:
    if chunking_style == "semantic":
        splitter = SemanticChunker(
            embeddings=OpenAIEmbeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        return splitter.split_documents(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
