# RAG from Scratch 🧠⚡

> Stop relying on black-box AI solutions. Build your own **Retrieval-Augmented Generation** system from the ground up — understand exactly how document retrieval, vector search, and LLM grounding work. Then take it further with **LoRA fine-tuning** to train your own custom model.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=chainlink)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20%7C%20Chroma-FF6B35)](https://faiss.ai)
[![Flask](https://img.shields.io/badge/API-Flask%20%2B%20SSE-000000?logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Is This?

Most RAG tutorials hand you a 5-line wrapper around a paid API. This project builds every layer **from scratch** so you actually understand what's happening:

| Layer | What you'll understand |
|---|---|
| **Document loading** | How raw files become queryable text |
| **Chunking** | Why overlap matters for retrieval quality |
| **Embeddings** | How meaning becomes a vector (locally, no API cost) |
| **Vector search** | How FAISS finds the "closest" meaning in milliseconds |
| **LangChain LCEL** | How retrieval + generation compose into a clean chain |
| **Streaming** | How tokens flow to the browser in real time via SSE |
| **Fine-tuning** | How LoRA adapts a pretrained LLM to your domain in hours |

---

## 🤖 The AI Agents (Core Pipeline)

Rather than relying on a single, monolithic LLM prompt, this system distributes cognitive workloads across specialized, autonomous AI components (Agents) to ensure high-accuracy, hallucination-free output. By separating concerns, each agent can be independently optimized, monitored, and scaled.

### 1. The Ingestion Agent (`src/ingest.py`) — *The Knowledge Builder*
Responsible for reading raw document streams, logically chunking them with semantic overlap, computing high-dimensional vector representations, and organizing them into a persistent vector database (FAISS/Chroma). This agent acts as the foundational memory encoder for the system.

### 2. The Retrieval & Reranking Agent (`src/retriever.py`) — *The Memory Engine*
When a user asks a query, this agent takes over. It performs a rapid top-k semantic vector search to find candidate knowledge blocks. It then employs a secondary Deep Learning model (a Cross-Encoder) to critically evaluate, re-score, and perfectly rank the retrieved context against the user's explicit intent.

### 3. The Generation Agent (`src/app.py`) — *The Communicator*
This agent synthesizes the perfectly ranked context provided by the Retrieval Agent with the user's instructions via a declarative LangChain LCEL pipe. It strictly forces the underlying LLM to ground its answers exclusively in the provided text, while simultaneously streaming tokens and citing its exact sources to the end-user in real-time.

### 4. The Domain-Adaptation Agents (`src/prepare_dataset.py`, `src/finetune.py`) — *The Trainers*
A sub-system of agents dedicated to continuous improvement. They autonomously analyze raw text corpuses to synthesize highly plausible Q&A training pairs (including adversarial, unanswerable scenarios). Furthermore, they handle the complex mathematics of injecting and training Low-Rank Adapters (LoRA) back into the base LLM weights, effectively teaching the model new, domain-specific expertise.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR DOCUMENTS                              │
│                   (PDF · TXT · DOCX · Markdown)                     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
          ┌───────────────────────────────┐
          │         STEP 1: INGEST        │  src/ingest.py
          │                               │
          │  Load → Split → Embed → Store │
          │                               │
          │  • RecursiveCharacterSplitter │  512 chars + 64 overlap
          │  • all-MiniLM-L6-v2 (local)  │  384-dim vectors, no API key
          │  • FAISS or Chroma on disk    │  persistent, fast
          └───────────────┬───────────────┘
                          │  Vector index stored on disk
                          │
          ┌───────────────▼───────────────┐
          │       STEP 2: RETRIEVE        │  src/retriever.py
          │                               │
          │  Query → Embed → Top-K search │
          │                               │
          │  • Same embedding model       │  apples-to-apples comparison
          │  • Cosine similarity search   │  semantic, not keyword
          │  • Returns chunks + scores    │  full transparency
          └───────────────┬───────────────┘
                          │  Grounded context
                          │
          ┌───────────────▼───────────────┐
          │       STEP 3: GENERATE        │  src/app.py
          │                               │
          │  Context + Question → LLM     │
          │                               │
          │  • LCEL chain (pipe syntax)   │  composable, swappable
          │  • Ollama (local) or OpenAI   │  your choice
          │  • Streams tokens via SSE     │  real-time UI response
          │  • Cites source documents     │  no hallucinations
          └───────────────┬───────────────┘
                          │  (optional domain adaptation)
                          │
          ┌───────────────▼───────────────┐
          │       STEP 4: FINE-TUNE       │  src/finetune.py
          │                               │
          │  Q&A Dataset → LoRA Training  │
          │                               │
          │  • Auto-generate training data│  from your own documents
          │  • LoRA: train only ~1% params│  fast, low memory
          │  • QLoRA: 4-bit quantization  │  fits on consumer GPU
          │  • TRL SFTTrainer             │  production-grade loop
          │  • Export adapter for Ollama  │  plug back into RAG
          └───────────────────────────────┘
```

---

## Features

- **Advanced Retrieval System** — Two-stage retrieval using FAISS for fast similarity search and `cross-encoder/ms-marco-MiniLM-L-6-v2` for high-accuracy semantic reranking.
- **Zero-cost embeddings** — All embedding and reranking models run 100% locally, requiring no API keys.
- **Dual LLM support** — Select between Ollama (free, local) or OpenAI effortlessly via the `.env` configuration.
- **Streaming UI** — Answers stream token-by-token with Server-Sent Events alongside real-time transparent source citations.
- **LoRA Fine-Tuning Engine** — Train a domain-specific LLM natively on a consumer GPU or CPU using TRL, complete with automated checkpoint resumption.
- **Autonomous Dataset Generation** — Auto-generates sophisticated Q&A training pairs (including negative/unanswerable scenarios) directly from your documents.
- **External Dataset Integration** — Built-in pipelines to automatically fetch and format massive public datasets (like SQuAD v2.0) into Alpaca JSONL formats for immediate fine-tuning.
- **Dark glassmorphism UI** — Modern, responsive drag-and-drop web interface.

---

## Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.ai)** (free, local LLM) — OR an OpenAI API key

### 2. Clone and install

```bash
git clone https://github.com/naveditachaudhary-pixel/rag-from-scratch.git
cd rag-from-scratch
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and choose your LLM:

```env
# Option A: Local (free, no API key)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2

# Option B: OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=sk-...
```

If using Ollama, pull a model:
```bash
ollama pull llama3.2
```

### 4. Run

```bash
python src/app.py
```

**Open [http://localhost:5050](http://localhost:5050)** in your browser.

---

## Usage

### Web UI (recommended)

1. Open `http://localhost:5050`
2. **Upload** your PDF, TXT, or DOCX files — or click **Load Sample** to try the built-in demo docs
3. Click **Index Documents** — watch the pipeline animate as chunks are embedded
4. **Ask anything** in the chat — answers stream in with source citations

### Command Line

**Ingest documents:**
```bash
python src/ingest.py --source ./data/my_docs --store faiss
python src/ingest.py --source ./report.pdf     --store chroma
```

**Query:**
```bash
python src/retriever.py --query "What is the attention mechanism?"
python src/retriever.py --query "Summarize the key findings"
```

---

## Fine-Tuning Your Own Model

Turn domain knowledge into model weights using LoRA. No cloud GPU required.

### Step 1 — Install fine-tuning dependencies

```bash
pip install -r requirements-finetune.txt
```

### Step 2 — Generate training data from your documents

```bash
# Auto-generates Q&A pairs using your configured LLM
python src/prepare_dataset.py \
  --source ./data/sample_docs \
  --output ./data/training/qa_pairs.jsonl \
  --num-questions 5
```

Or write your own pairs manually in JSONL format:
```jsonl
{"instruction": "What is the refund policy?", "input": "", "output": "Refunds are accepted within 30 days..."}
{"instruction": "How do I reset my password?", "input": "", "output": "Click 'Forgot Password' on the login page..."}
```

### Step 3 — Fine-tune

```bash
# CPU (slow but works everywhere) — great for testing
python src/finetune.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset ./data/training/sample_qa.jsonl \
  --epochs 3

# GPU recommended — Phi-3-mini with 4-bit QLoRA (~15 min on RTX 3060)
python src/finetune.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --quantize \
  --epochs 3
```

The LoRA adapter is saved to `./models/lora-adapter/`.

### Step 4 — Use your fine-tuned model

```bash
# Test it immediately
python src/finetune.py --test-only --test-q "Explain the refund policy"

# The script also prints Ollama export instructions — plug it back into the RAG system!
```

### Hardware Guide

| Hardware | Recommended setup | ~Time per epoch |
|---|---|---|
| CPU only | TinyLlama, batch 1, no `--quantize` | 2–4 hours |
| GPU 6GB VRAM | Phi-3-mini + `--quantize` (QLoRA) | 15–30 min |
| GPU 12GB VRAM | Phi-3-mini full LoRA, or Llama-3.2-3B | 10–20 min |
| GPU 24GB+ VRAM | Any model, no quantization needed | 5–10 min |

---

## Project Structure

```
rag-from-scratch/
│
├── src/
│   ├── ingest.py              # Step 1: Load → Chunk → Embed → Store
│   ├── retriever.py           # Step 2: Query → Retrieve → LangChain LCEL chain
│   ├── app.py                 # Step 3: Flask REST API + SSE streaming
│   ├── prepare_dataset.py     # Step 4a: Generate Q&A training data from docs
│   └── finetune.py            # Step 4b: LoRA/QLoRA fine-tuning with TRL+PEFT
│
├── static/
│   ├── css/style.css          # Dark glassmorphism design system
│   └── js/app.js              # Frontend: drag-drop, SSE streaming, UI logic
│
├── templates/
│   └── index.html             # Single-page web UI
│
├── data/
│   ├── sample_docs/           # Built-in demo documents (RAG + LangChain guides)
│   └── training/              # Training datasets go here
│       └── sample_qa.jsonl    # 12 pre-written Q&A pairs for instant fine-tuning
│
├── models/                    # Fine-tuned LoRA adapters saved here (auto-created)
├── vectorstore/               # FAISS index saved here after ingestion (auto-created)
│
├── .env                       # Your config (gitignored — never commit secrets)
├── .env.example               # Config template with all options documented
├── requirements.txt           # Core RAG dependencies
└── requirements-finetune.txt  # Fine-tuning dependencies (PEFT, TRL, datasets)
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | System health: store ready, LLM config, embed model |
| `POST` | `/api/ingest` | Upload files (multipart) or use `{"use_sample": true}` |
| `POST` | `/api/query` | Ask a question — returns `{answer, sources, query}` |
| `GET` | `/api/query/stream?q=...` | Streaming answer via SSE — tokens arrive in real time |

### Example: POST /api/query

```bash
curl -X POST http://localhost:5050/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS and how does it compare to Chroma?"}'
```

```json
{
  "success": true,
  "query": "What is FAISS and how does it compare to Chroma?",
  "answer": "FAISS is Facebook's Similarity Search library optimized for in-memory vector search...",
  "sources": [
    {
      "source": "rag_concepts.txt",
      "page": null,
      "content": "FAISS (Facebook AI Similarity Search) is an open-source library..."
    }
  ]
}
```

---

## Configuration Reference

All settings live in `.env`:

```env
# LLM
LLM_PROVIDER=ollama          # 'ollama' or 'openai'
LLM_MODEL=llama3.2           # any Ollama model or OpenAI model name
OPENAI_API_KEY=sk-...        # only needed if LLM_PROVIDER=openai

# Embeddings (local, no key needed)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store
VECTOR_STORE=faiss           # 'faiss' or 'chroma'
VECTOR_STORE_PATH=./vectorstore

# Chunking
CHUNK_SIZE=512               # characters per chunk
CHUNK_OVERLAP=64             # overlap between consecutive chunks
TOP_K_RESULTS=4              # chunks retrieved per query

# Server
FLASK_PORT=5050
FLASK_DEBUG=false

# Fine-Tuning
FINETUNE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
HUGGINGFACE_TOKEN=hf_...     # only for gated models (Llama 3.2)
LORA_OUTPUT_DIR=./models/lora-adapter
```

---

## Key Concepts Explained

**Why chunking overlap matters:**
Without overlap, a sentence that straddles a chunk boundary gets split — one chunk has the beginning, the next has the end. With 64-char overlap, both chunks contain the full sentence, so retrieval always captures complete ideas.

**Why local embeddings work so well:**
`all-MiniLM-L6-v2` was trained on 1B+ sentence pairs. It maps semantically similar text to nearby points in 384-dimensional space. "automobile" and "car" are neighbors; "bank account" and "river bank" are far apart. No API call, no latency, no cost.

**Why LoRA instead of full fine-tuning:**
A 1B-parameter model has ~4GB of weights. Full fine-tuning updates every parameter — needs 40GB+ VRAM. LoRA injects tiny matrices (rank 16) into each attention layer and trains only those — ~2M parameters instead of 1B. Same quality improvements, fraction of the hardware.

**Why SSE instead of WebSockets:**
Server-Sent Events are one-directional (server → client) and require zero JavaScript libraries. They reconnect automatically and work through most proxies. For streaming LLM output, SSE is simpler and more reliable than WebSockets.

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your feature branch: `git checkout -b feat/my-feature`
3. Commit your changes: `git commit -m 'feat: add my feature'`
4. Push to the branch: `git push origin feat/my-feature`
5. Open a Pull Request

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

*Built to learn, built to ship.*
