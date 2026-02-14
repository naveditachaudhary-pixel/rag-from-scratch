"""
RAG from Scratch — Step 3: Production Flask API
=================================================
This module provides:
  • REST API endpoints for ingestion and querying
  • Server-Sent Events (SSE) for streaming LLM responses
  • Static file serving for the web UI

Endpoints:
  POST /api/ingest         — Upload & index documents
  POST /api/query          — Ask a question, get an answer + sources
  GET  /api/query/stream   — Streaming response via SSE
  GET  /api/status         — Check if vector store exists
  GET  /                   — Serve the web UI
"""

import os
import sys
import json
import time
import logging
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, Response, send_from_directory, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from ingest import ingest
from retriever import load_vector_store, get_llm, query_rag, build_rag_chain

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App Config ────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR   = BASE_DIR / "static"
UPLOAD_DIR   = BASE_DIR / "data"
STORE_PATH   = os.getenv("VECTOR_STORE_PATH", str(BASE_DIR / "vectorstore"))
STORE_TYPE   = os.getenv("VECTOR_STORE", "faiss")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
CORS(app)

# ── Cached singletons (loaded once on first request) ──────────────────────────
_vectorstore = None
_llm         = None


def get_store():
    global _vectorstore
    if _vectorstore is None:
        if not Path(STORE_PATH).exists():
            return None
        _vectorstore = load_vector_store(
            store_type=STORE_TYPE,
            persist_path=STORE_PATH,
            embed_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
    return _vectorstore


def get_cached_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def reset_store_cache():
    global _vectorstore
    _vectorstore = None


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def status():
    """Check if the vector store has been built."""
    store_exists = Path(STORE_PATH).exists() and any(Path(STORE_PATH).iterdir())
    data_files   = []
    if UPLOAD_DIR.exists():
        data_files = [
            f.name for f in UPLOAD_DIR.iterdir()
            if f.suffix.lower() in ALLOWED_EXTENSIONS
        ]

    return jsonify({
        "store_ready": store_exists,
        "store_type":  STORE_TYPE,
        "store_path":  STORE_PATH,
        "data_files":  data_files,
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "llm_model":   os.getenv("LLM_MODEL", "llama3.2"),
        "embed_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    })


@app.route("/api/ingest", methods=["POST"])
def ingest_documents():
    """
    Upload documents and build the vector store.
    Accepts multipart/form-data with one or more files.
    Also accepts JSON {"use_sample": true} to use the built-in sample data.
    """
    try:
        # Handle sample data flag
        if request.is_json:
            data = request.get_json()
            if data.get("use_sample"):
                sample_path = BASE_DIR / "data" / "sample_docs"
                if not sample_path.exists():
                    return jsonify({"error": "Sample data not found. Create ./data/sample_docs/"}), 404
                result = ingest(str(sample_path), STORE_TYPE, STORE_PATH)
                reset_store_cache()
                return jsonify({"success": True, **result})

        # Handle file uploads
        if "files" not in request.files:
            return jsonify({"error": "No files provided. Send files via 'files' field."}), 400

        uploaded_files = request.files.getlist("files")
        if not uploaded_files:
            return jsonify({"error": "No files selected."}), 400

        # Save uploaded files
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                return jsonify({
                    "error": f"Unsupported file type: {suffix}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                }), 400
            save_path = UPLOAD_DIR / filename
            file.save(str(save_path))
            saved_paths.append(str(save_path))
            logger.info(f"📥 Saved uploaded file: {save_path}")

        # Ingest from the data directory
        result = ingest(str(UPLOAD_DIR), STORE_TYPE, STORE_PATH)
        reset_store_cache()

        return jsonify({
            "success": True,
            "uploaded_files": [Path(p).name for p in saved_paths],
            **result,
        })

    except Exception as e:
        logger.exception("Ingestion failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query():
    """
    Query the RAG system.
    Body: {"query": "your question here"}
    """
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Request body must contain 'query' field."}), 400

        question = data["query"].strip()
        if not question:
            return jsonify({"error": "Query cannot be empty."}), 400

        store = get_store()
        if store is None:
            return jsonify({
                "error": "Vector store not found. Please ingest documents first via POST /api/ingest"
            }), 404

        llm = get_cached_llm()
        result = query_rag(question, store, llm, stream=False)

        return jsonify({
            "success":  True,
            "query":    result["query"],
            "answer":   result["answer"],
            "sources":  result["sources"],
        })

    except Exception as e:
        logger.exception("Query failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/query/stream", methods=["GET"])
def query_stream():
    """
    Streaming query via Server-Sent Events (SSE).
    Usage: GET /api/query/stream?q=your+question
    """
    question = request.args.get("q", "").strip()
    if not question:
        return jsonify({"error": "Provide question via ?q= parameter"}), 400

    store = get_store()
    if store is None:
        def error_stream():
            yield f"data: {json.dumps({'error': 'No vector store. Ingest documents first.'})}\n\n"
        return Response(error_stream(), mimetype="text/event-stream")

    def generate():
        try:
            llm = get_cached_llm()
            chain, retriever = build_rag_chain(store, llm)

            # Send sources first
            docs = retriever.invoke(question)
            sources = [
                {
                    "content": doc.page_content[:250],
                    "source":  doc.metadata.get("source", "Unknown"),
                    "page":    doc.metadata.get("page", None),
                }
                for doc in docs
            ]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Stream the LLM response token by token
            for token in chain.stream(question):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.exception("Streaming query failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.getenv("FLASK_PORT", 5050))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"🚀 RAG API server starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
