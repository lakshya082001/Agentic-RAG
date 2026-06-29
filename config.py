import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    base_url=os.environ.get("OPENAI_API_BASE"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

embedder = OpenAIEmbeddings(
    model=os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
    base_url=os.environ.get("OPENAI_API_BASE"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)
