# FinReg Knowledge Graph — AI Regulatory Research Assistant

A knowledge graph chatbot for researching U.S. financial regulation documents focused on AI governance and compliance.

## What It Does

Financial regulators (FINRA, SEC, CFTC, Federal Reserve, NIST) are producing guidance on AI use in financial services at a rapid pace. This tool lets you upload those documents, automatically extracts the key entities and concepts, builds a graph of how they relate across documents, and lets you ask natural-language questions grounded in the actual source material.

The system answers questions by finding the most relevant slice of the knowledge graph for your query and passing it as structured context to Claude — so responses cite specific documents and explain connections across regulatory frameworks.

## Key Capabilities

- **Cross-document entity linking** — the same organization, regulation, or concept appearing across multiple PDFs gets connected in the graph automatically
- **Semantic similarity edges** — concepts that are related in meaning (even if phrased differently) get linked via sentence embeddings
- **Graph-grounded answers** — the LLM sees a structured subgraph, not a raw text blob, so it can explain *why* sources are connected
- **Interactive graph visualization** — explore entity relationships visually via the left-panel network graph
- **Persistent graphs** — download the graph as a `.pkl` file and reload it in a future session

## Pre-loaded Documents

The `finreg-kg/data/` folder ships with:

| Document | Source |
|----------|--------|
| FINRA 2026 Annual Regulatory Oversight Report | FINRA |
| FINRA 2025 Annual Regulatory Oversight Report | FINRA |
| FINRA AI in the Securities Industry (2020) | FINRA |
| CFTC Staff Advisory No. 24-17 — AI in Derivatives Markets | CFTC |
| Federal Reserve / OCC / FDIC / CFPB / NCUA Joint RFI on AI | Interagency |
| NIST AI Risk Management Framework 1.0 | NIST |

## Stack

- **Streamlit** — UI
- **spaCy** (`en_core_web_lg`) — named entity recognition
- **NetworkX** — in-memory knowledge graph
- **sentence-transformers** (`all-MiniLM-L6-v2`) — semantic similarity
- **PyVis** — interactive graph visualization
- **PyMuPDF** — PDF text extraction
- **Anthropic Claude** (`claude-haiku-4-5-20251001`) — LLM reasoning

## Quickstart

```bash
cd finreg-kg
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_lg
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Add your Anthropic API key to .streamlit/secrets.toml
.venv/bin/streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). The pre-loaded documents will be ingested automatically on first load.
