# 🧠 Second Brain — Personal Language Model

> A privacy-first AI that learns from your own data and answers questions *as you* — 
> built to demonstrate the core vision of [Personal AI](https://personal.ai)'s Personal Language Model (PLM).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Runs Locally](https://img.shields.io/badge/Runs-Fully%20Local-green)
![No GPU](https://img.shields.io/badge/GPU-Not%20Required-lightgrey)

---

## 🎯 What This Is

Most AI assistants know everything about the world but nothing about *you*.  
**Second Brain** flips that: it knows only what you tell it, and answers every question 
grounded entirely in your personal data — documents, notes, URLs, and writings.

This is a working prototype of a **Personal Language Model (PLM)** — the core concept 
behind Personal AI's platform.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Memory Transparency** | Every answer shows *which* memories were used, with relevance scores |
| **Privacy-First** | All data stored locally in Qdrant — only the query goes to the LLM API |
| **Staleness Detection** | Older memories get lower trust weight; flagged in the UI |
| **Multi-Source Ingestion** | PDF, DOCX, TXT, Markdown, URLs, plain text |
| **CPU-Only** | Runs on a laptop with no GPU using quantized sentence-transformers |
| **Confidence Scoring** | Every answer rated High / Medium / Low / None based on memory match quality |

---

## 🏗 Architecture

```
Personal Data (PDF, DOCX, TXT, URLs, Notes)
         │
         ▼
┌─────────────────────┐
│  Ingestion Pipeline │  loaders.py → chunker.py → embedder.py
│  Smart chunking by  │
│  paragraph boundary │
└─────────┬───────────┘
          │  384-dim vectors (all-MiniLM-L6-v2)
          ▼
┌─────────────────────┐
│   Qdrant Vector DB  │  Runs locally, no cloud needed
│   (Local Mode)      │  Stores chunks + metadata + embeddings
└─────────┬───────────┘
          │  Semantic search (cosine similarity)
          ▼
┌─────────────────────┐
│    RAG Pipeline     │  Retrieves top-K memories → builds persona prompt
│                     │  → calls LLM API → returns answer + memory citations
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Streamlit UI      │  Chat interface + Memory Transparency Panel
│                     │  Shows which memories influenced each answer
└─────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourname/second-brain
cd second-brain
pip install -r requirements.txt
```

### 2. Set Up API Key
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY
```

### 3. Add Your First Memories
```bash
# Add a PDF
python ingest.py --file my_resume.pdf

# Add a directory of notes
python ingest.py --dir ./my_notes/

# Add a URL
python ingest.py --url https://yourwebsite.com/about

# Add a quick text note
python ingest.py --text "I am a data engineer with 5 years of experience in Python and Airflow"
```

### 4. Launch the App
```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` and start asking questions.

---

## 💬 Example Queries

Once you've loaded your data, try:
- *"What are my main technical skills?"*
- *"What projects have I worked on at [company]?"*
- *"What do I think about [topic]?"*
- *"Summarize my professional background"*
- *"What have I written about machine learning?"*

---

## 🔍 Memory Transparency Panel

The right-side panel shows for every answer:
- **Which documents** the memories came from
- **Relevance score** (cosine similarity to your query)
- **Trust weight** (relevance × staleness factor)
- **Chunk position** in the original document
- **Date** the memory was added
- **Staleness warning** for older memories

This is the core philosophical differentiator from generic AI: 
**you can always see exactly why the AI said what it said.**

---

## 📁 Project Structure

```
second-brain/
├── ingestion/
│   ├── loaders.py      # PDF, DOCX, TXT, URL, raw text loading
│   ├── chunker.py      # Smart paragraph-aware chunking
│   └── embedder.py     # sentence-transformers embedding (CPU)
├── memory/
│   └── store.py        # Qdrant vector DB wrapper + staleness scoring
├── rag/
│   └── pipeline.py     # RAG chain: retrieve → prompt → generate
├── ui/
│   └── app.py          # Streamlit chat UI + memory transparency panel
├── data/
│   └── qdrant_db/      # Local vector database (gitignored)
├── ingest.py           # CLI ingestion tool
├── requirements.txt
└── .env.example
```

---

## 🛠 Tech Stack

| Component | Tool | Why |
|---|---|---|
| Embedding | `all-MiniLM-L6-v2` | 384-dim, fast on CPU, excellent quality |
| Vector DB | Qdrant (local) | No server, privacy-first, production-grade |
| LLM | Anthropic Claude / OpenAI | Swap via `.env` config |
| UI | Streamlit | Fast to build, easy to demo |
| Ingestion | PyPDF2, python-docx, BeautifulSoup | Cover all personal data formats |

---

## 🔒 Privacy Design

- **Your data never leaves your machine** except for the query text sent to the LLM API
- Vector embeddings computed locally using sentence-transformers
- Qdrant runs in local file mode — no external connections
- LLM only receives the query + retrieved text snippets (never your full documents)

---

## 📬 Contact

Built as a portfolio project demonstrating the Personal Language Model concept.  
Inspired by [Personal AI](https://personal.ai)'s vision of user-controlled, memory-grounded AI.
