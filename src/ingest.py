"""
RAG from Scratch — Step 1: Document Ingestion & Vector Store Builder
=====================================================================
This module handles:
  • Loading documents (PDF, TXT, DOCX, or raw text)
  • Splitting them into chunks with overlap
  • Generating embeddings via a local HuggingFace model
  • Persisting to FAISS or Chroma vector store

Usage:
    python src/ingest.py --source ./data --store faiss
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm

# LangChain document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1A: Load Documents
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(source_path: str) -> List[Document]:
    """
    Load documents from a directory or a single file.
    Supports: .pdf, .txt, .md, .docx
    """
    path = Path(source_path)
    docs: List[Document] = []

    if path.is_dir():
        logger.info(f"📂 Loading all documents from directory: {source_path}")
        loaders = {
            "**/*.pdf":  PyPDFLoader,
            "**/*.txt":  TextLoader,
            "**/*.md":   TextLoader,
            "**/*.docx": Docx2txtLoader,
        }
        for glob_pattern, loader_cls in loaders.items():
            dir_loader = DirectoryLoader(
                source_path,
                glob=glob_pattern,
                loader_cls=loader_cls,
                show_progress=True,
                silent_errors=True,
            )
            loaded = dir_loader.load()
            docs.extend(loaded)
            logger.info(f"  ↳ {glob_pattern}: {len(loaded)} document(s) loaded")

    elif path.is_file():
        suffix = path.suffix.lower()
        logger.info(f"📄 Loading single file: {source_path}")
        if suffix == ".pdf":
            docs = PyPDFLoader(source_path).load()
        elif suffix in (".txt", ".md"):
            docs = TextLoader(source_path, encoding="utf-8").load()
        elif suffix == ".docx":
            docs = Docx2txtLoader(source_path).load()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    else:
        raise FileNotFoundError(f"Source not found: {source_path}")

    logger.info(f"✅ Total documents loaded: {len(docs)}")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1B: Split into Chunks
# ─────────────────────────────────────────────────────────────────────────────

def split_documents(docs: List[Document], chunk_size: int = 512, chunk_overlap: int = 64) -> List[Document]:
    """
    Split documents into overlapping chunks for better retrieval accuracy.
    
    WHY OVERLAP?
    Without overlap, a sentence split across two chunks may lose context.
    With overlap (e.g. 64 tokens), consecutive chunks share a boundary,
    so relevant passages are always fully captured.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"✂️  Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1C: Create Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Return a local HuggingFace embedding model.
    
    WHY LOCAL EMBEDDINGS?
    • No API key needed, no cost, no data leaves your machine
    • all-MiniLM-L6-v2 is tiny (~90MB) but powerful — great for most use cases
    • Embeddings are 384-dimensional dense vectors
    """
    logger.info(f"🤖 Loading embedding model: {model_name} (may download ~90MB on first run)")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1D: Build & Persist Vector Store
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    store_type: str = "faiss",
    persist_path: str = "./vectorstore",
) -> None:
    """
    Embed all chunks and store them in FAISS or Chroma.
    
    FAISS  → Facebook AI Similarity Search — blazing fast, in-memory/on-disk
    Chroma → Persistent, queryable DB with metadata filtering support
    """
    os.makedirs(persist_path, exist_ok=True)

    if store_type == "faiss":
        logger.info("🗄️  Building FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(persist_path)
        logger.info(f"💾 FAISS index saved to: {persist_path}")

    elif store_type == "chroma":
        logger.info("🗄️  Building Chroma database...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_path,
        )
        logger.info(f"💾 Chroma DB saved to: {persist_path}")

    else:
        raise ValueError(f"Unknown store type: {store_type}. Choose 'faiss' or 'chroma'.")

    logger.info(f"✅ Vector store built with {len(chunks)} chunks!")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def ingest(source: str, store_type: str = None, persist_path: str = None) -> dict:
    """Full ingestion pipeline — callable from Python or CLI."""
    store_type   = store_type   or os.getenv("VECTOR_STORE", "faiss")
    persist_path = persist_path or os.getenv("VECTOR_STORE_PATH", "./vectorstore")
    chunk_size   = int(os.getenv("CHUNK_SIZE", 512))
    chunk_overlap= int(os.getenv("CHUNK_OVERLAP", 64))
    embed_model  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    docs   = load_documents(source)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    emb    = get_embedding_model(embed_model)
    build_vector_store(chunks, emb, store_type, persist_path)

    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "store": store_type,
        "path": persist_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    parser.add_argument("--source", default="./data", help="Path to documents folder or file")
    parser.add_argument("--store",  default=None,     help="Vector store: 'faiss' or 'chroma'")
    parser.add_argument("--path",   default=None,     help="Where to persist the vector store")
    args = parser.parse_args()

    result = ingest(args.source, args.store, args.path)
    print("\n" + "="*50)
    print("🎉 Ingestion Complete!")
    print(f"   Documents : {result['documents']}")
    print(f"   Chunks    : {result['chunks']}")
    print(f"   Store     : {result['store']}")
    print(f"   Saved to  : {result['path']}")
    print("="*50)
