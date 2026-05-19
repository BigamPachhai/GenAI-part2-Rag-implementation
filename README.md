# RAG Book Assistant

A simple Retrieval-Augmented Generation (RAG) demo that builds a Chroma vector store from a PDF and answers questions using Mistral with free/local embeddings.

## What’s included

- `create_database.py` — builds a persistent Chroma DB from a PDF.
- `main.py` — CLI prompt for asking questions using the stored vectors.
- `app.py` — Streamlit UI for upload, indexing, and Q&A.

## Setup

1. Create a `.env` file in the project root (or copy from `.env.example`) and add your API keys:

```
MISTRAL_API_KEY=your_mistral_key
```

2. Ensure your PDF exists at `document loaders/deeplearning.pdf` or update the path in `create_database.py`.

> The Mistral API key is only needed for answering questions (`main.py` or `app.py`). Building the vector DB does not call any paid API.

## Run

- Build the vector DB:
  - `create_database.py`
- Ask questions (CLI):
  - `main.py`
- Or use the UI:
  - `app.py`

## Notes

- The vector DB is saved to `chroma_db_hf/`.
- If `chroma_db_hf/` is missing, run the database creation step first.
