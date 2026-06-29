# Agentic RAG

A modular Retrieval-Augmented Generation pipeline with an agentic LangGraph workflow and a Gradio chat UI.

## Architecture

```
src/
├── ingestion.py    # URL + PDF + raw-text loading
├── chunking.py     # Recursive or semantic chunking
├── vectorstore.py  # Chroma, FAISS, or raw ChromaDB collection
├── retrieval.py    # MMR retrieval (LangChain + custom implementation)
├── pipeline.py     # End-to-end RAG pipeline function
└── workflow.py     # LangGraph agent + retrieval tool
config.py           # LLM and embedder initialisation
app.py              # Gradio chat interface (entry point)
```

## Setup

```bash
cp .env.example .env   # fill in your keys
uv venv
uv pip install -r requirements.txt
```

## Run

```bash
uv run python app.py
```

To stop: `Ctrl+C`. To restart, run the same command again.

## Data

Place PDF files in the `Data/` folder. They are gitignored and never committed.

## Configuration

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `OPENAI_API_BASE` | API base URL (defaults to OpenAI) |
| `LLM_MODEL` | Chat model name (default: `gpt-4o-mini`) |
| `EMBEDDING_MODEL` | Embedding model name (default: `text-embedding-ada-002`) |

## Vectorstore options

Pass `store_type` to `rag_pipeline()` or `create_vectorstore()`:

- `"faiss"` — FAISS with HNSW index (default)
- `"chroma"` — LangChain Chroma wrapper (persisted to `./db`)
- anything else — raw ChromaDB collection with custom MMR scoring
