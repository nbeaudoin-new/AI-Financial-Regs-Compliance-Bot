# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
cd finreg-kg
.venv/bin/streamlit run app.py
```

The app runs on Python 3.13. The virtualenv is at `finreg-kg/.venv`.

## First-Time Setup

```bash
cd finreg-kg
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_lg
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Add your ANTHROPIC_API_KEY to .streamlit/secrets.toml
```

## Architecture

This is a graph-based RAG system for financial regulation research. The pipeline is:

**PDF → Extraction → Knowledge Graph → Semantic Indexing → Subgraph Retrieval → LLM Answer**

### Module Responsibilities

| Module | Role |
|--------|------|
| `src/extractor.py` | PyMuPDF PDF → `{filename, pages, full_text}` dict |
| `src/graph_builder.py` | spaCy NER + noun chunks → NetworkX graph with DOCUMENT/ENTITY/CONCEPT nodes |
| `src/graph_store.py` | Pickle serialization for graph download/upload |
| `src/retriever.py` | Sentence-transformer embeddings for semantic edges + cosine-similarity subgraph retrieval |
| `src/llm.py` | Formats subgraph context and calls Claude with chat history |
| `app.py` | Streamlit UI — two-column layout (graph viz + chat), auto-loads `data/` on startup |

### Graph Schema

- **Node types** (`node_type` attr): `DOCUMENT`, `ENTITY`, `CONCEPT`
- **Edge types** (`rel` attr): `MENTIONS` (doc→entity/concept), `CO_OCCURS` (entity↔entity in same sentence, weighted), `RELATED_TO` (semantic similarity ≥ 0.75)
- spaCy entity labels captured: `ORG`, `LAW`, `GPE`, `PERSON`, `DATE`, `PRODUCT`
- Text truncated to 100k chars per doc for spaCy performance

### Key Behaviors

- `@st.cache_resource` on both the spaCy model (`load_spacy_model`) and sentence-transformer (`load_sentence_transformer`) — never reload these
- PDFs in `data/` are auto-ingested on first page load via `autoload_data_folder()` (guarded by `st.session_state.data_loaded`)
- `st.chat_input` must remain at the top level of `app.py`, not inside a column block — Streamlit requires this
- Graph mutations always update `st.session_state.graph` and reassign `st.session_state.retriever`
- LLM model: `claude-haiku-4-5-20251001`

### Data Folder

Pre-loaded regulatory PDFs live in `finreg-kg/data/`. Drop additional PDFs there to include them on next startup, or use the in-app uploader to add them at runtime.
