# RAG from Scratch

> Build your own Retrieval-Augmented Generation system — no black boxes. Understand how grounding, retrieval, and fine-tuning actually work.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-green)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS%20%7C%20Chroma-orange)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Architecture

```
Documents (PDF / TXT / DOCX)
        │
        ▼
 ┌─────────────────┐
 │  1. INGEST      │  Load → Chunk → Embed → Store
 │  src/ingest.py  │  (offline, run once)
 └────────┬────────┘
          │  FAISS / Chroma vector index
          ▼
 ┌─────────────────┐
 │  2. RETRIEVE    │  Query → Embed → Similarity Search → Top-K Chunks
 │  src/retriever.py
 └────────┬────────┘
          │  Grounded context
          ▼
 ┌─────────────────┐
 │  3. GENERATE    │  [Context + Question] → LLM → Grounded Answer
 │  src/app.py     │  (OpenAI or local Ollama)
 └─────────────────┘
          │  (optional)
          ▼
 ┌─────────────────┐
 │  4. FINE-TUNE   │  Q&A Dataset → LoRA Training → Custom LLM Adapter
 │  src/finetune.py│
 └─────────────────┘
```

---

## Tech Stack

| Component       | Library                          | Notes                          |
|-----------------|----------------------------------|--------------------------------|
| Orchestration   | LangChain 1.x + LCEL             | Pipe-based chain composition   |
| Vector Store    | FAISS / Chroma                   | Swappable via `.env`           |
| Embeddings      | `all-MiniLM-L6-v2` (local)      | No API key, 90MB, offline      |
| LLM             | Ollama (local) or OpenAI         | Configurable in `.env`         |
| Fine-Tuning     | TRL + PEFT (LoRA/QLoRA)          | Works on consumer GPU or CPU   |
| Web API         | Flask + Server-Sent Events       | Streaming token-by-token       |
| Frontend        | Vanilla HTML / CSS / JS          | No framework dependencies      |

---

## Quick Start (3 Steps)

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed locally **OR** an OpenAI API key

### Step 1 — Install

```bash
pip install -r requirements.txt
```

### Step 2 — Configure

```bash
cp .env.example .env
# Edit .env: set LLM_PROVIDER to 'ollama' or 'openai'
```

If using Ollama:
```bash
ollama pull llama3.2
```

### Step 3 — Run

```bash
python src/app.py
# Open http://localhost:5050
```

Upload your documents, click **Index**, then ask questions!

---

## Fine-Tuning Your Own Model

Add domain expertise directly into the model weights using LoRA — no API key needed.

### Step 1 — Install fine-tuning dependencies

```bash
pip install -r requirements-finetune.txt
```

### Step 2 — Prepare your training data

**Option A — Auto-generate from your documents:**
```bash
# Uses your LLM to generate Q&A pairs from document chunks
python src/prepare_dataset.py \
  --source ./data/sample_docs \
  --output ./data/training/qa_pairs.jsonl \
  --num-questions 5
```

**Option B — Write your own JSONL:**
```json
{"instruction": "What is the company refund policy?", "input": "", "output": "Refunds are accepted within 30 days of purchase..."}
```

### Step 3 — Fine-tune with LoRA

```bash
# CPU mode (slow but works anywhere) — TinyLlama, 3 epochs
python src/finetune.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset ./data/training/sample_qa.jsonl \
  --epochs 3

# GPU mode (recommended) — Phi-3-mini with 4-bit QLoRA
python src/finetune.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --quantize \
  --epochs 3
```

The LoRA adapter is saved to `./models/lora-adapter/`.

### Step 4 — Use your fine-tuned model

```bash
# Test inference directly
python src/finetune.py --test-only --test-q "Explain the refund policy"

# Or export to Ollama and use it in the RAG system
# (instructions printed automatically after training)
```

### Hardware Guide

| Setup              | Recommended Config                          | Speed          |
|--------------------|---------------------------------------------|----------------|
| No GPU (CPU only)  | TinyLlama, `--batch-size 1`, no `--quantize`| ~2-4 hrs/epoch |
| GPU 6GB+ (VRAM)    | Phi-3-mini + `--quantize` (4-bit QLoRA)    | ~15-30 min     |
| GPU 12GB+ (VRAM)   | Phi-3-mini full LoRA, or Llama-3.2-3B      | ~10-20 min     |
| GPU 24GB+ (VRAM)   | Any model, no quantization needed           | ~5-10 min      |

---

## Project Structure

```
rag-from-scratch/
├── src/
│   ├── ingest.py           # Step 1: Load → Chunk → Embed → Store
│   ├── retriever.py        # Step 2: Semantic search + LangChain LCEL chain
│   ├── app.py              # Step 3: Flask API + SSE streaming
│   ├── prepare_dataset.py  # Step 4a: Auto-generate Q&A training data
│   └── finetune.py         # Step 4b: LoRA/QLoRA fine-tuning with TRL
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
├── data/
│   ├── sample_docs/        # Built-in demo documents
│   └── training/           # Training datasets (JSONL)
│       └── sample_qa.jsonl
├── models/                 # Fine-tuned LoRA adapters saved here
├── vectorstore/            # Auto-created after ingestion
├── .env.example
├── requirements.txt
└── requirements-finetune.txt
```

---

## API Reference

| Method | Endpoint               | Description                      |
|--------|------------------------|----------------------------------|
| GET    | `/api/status`          | System status + config           |
| POST   | `/api/ingest`          | Upload + index documents         |
| POST   | `/api/query`           | Synchronous Q&A                  |
| GET    | `/api/query/stream?q=` | Streaming Q&A via SSE            |

---

## Key Concepts

- **Chunking with overlap** — preserves sentence boundaries across chunk edges  
- **Local embeddings** — `all-MiniLM-L6-v2` runs 100% offline  
- **LCEL chain** — `retriever | format_docs | prompt | llm | StrOutputParser()`  
- **SSE streaming** — tokens flow to the browser as they're generated  
- **LoRA** — trains <1% of model parameters by injecting low-rank matrix pairs  
- **QLoRA** — adds 4-bit base model quantization for consumer GPU fine-tuning  

---

## License

MIT
