import glob
from langchain_community.document_loaders import UnstructuredURLLoader, PyMuPDFLoader
from langchain_core.documents import Document


def read_sources(urls: list[str], pdfs_path: str, raw_texts: list[str] = []) -> list[Document]:
    docs = []

    if urls:
        loader = UnstructuredURLLoader(urls=urls)
        docs.extend(loader.load())

    for file in glob.glob(f"{pdfs_path}/*.pdf"):
        loader = PyMuPDFLoader(file)
        docs.extend(loader.load())

    docs.extend([Document(page_content=t, metadata={"source": "manual"}) for t in raw_texts])

    return docs
