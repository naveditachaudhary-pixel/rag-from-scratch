"""
RAG from Scratch — Step 2: Retrieval Chain
============================================
This module handles:
  • Loading the persisted vector store
  • Semantic search / retrieval of relevant chunks
  • Sending retrieved context + question to the LLM
  • Streaming responses with source citations

Usage:
    python src/retriever.py --query "What is attention mechanism?"
"""

import os
import logging
from typing import Generator, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2A: Load Vector Store
# ─────────────────────────────────────────────────────────────────────────────

def load_vector_store(
    store_type: str = "faiss",
    persist_path: str = "./vectorstore",
    embed_model: str = "all-MiniLM-L6-v2",
):
    """Load a previously built FAISS or Chroma vector store from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if store_type == "faiss":
        logger.info(f"📦 Loading FAISS index from: {persist_path}")
        store = FAISS.load_local(
            persist_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    elif store_type == "chroma":
        logger.info(f"📦 Loading Chroma DB from: {persist_path}")
        store = Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings,
        )
    else:
        raise ValueError(f"Unknown store: {store_type}")

    logger.info("✅ Vector store loaded successfully")
    return store


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2B: Semantic Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks(store, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
    """
    Find the top-K most semantically similar chunks to the query.
    
    HOW IT WORKS:
    1. The query is embedded into a 384-dim vector (same embedding model as ingestion)
    2. Cosine similarity is computed against all stored chunk vectors
    3. Top-K closest vectors are returned with their similarity scores
    
    This is the CORE of RAG — grounding the LLM in your actual documents.
    """
    results = store.similarity_search_with_score(query, k=top_k)
    logger.info(f"🔍 Retrieved {len(results)} chunks for query: '{query[:60]}...'")
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "unknown")
        logger.info(f"   [{i+1}] score={score:.4f} | source={source}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2C: Build LLM
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(provider: str = None, model: str = None):
    """
    Return an LLM based on the configured provider.
    Supports: openai, ollama (local)
    """
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    model    = model    or os.getenv("LLM_MODEL", "llama3.2")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-your"):
            raise ValueError(
                "❌ OpenAI API key not set. Either:\n"
                "   • Set OPENAI_API_KEY in .env, OR\n"
                "   • Switch to Ollama: LLM_PROVIDER=ollama in .env"
            )
        logger.info(f"🧠 Using OpenAI model: {model}")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0.1)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        logger.info(f"🦙 Using Ollama model: {model} (local)")
        return ChatOllama(model=model, temperature=0.1)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'ollama'.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2D: The RAG Prompt Template
# ─────────────────────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable assistant that answers questions based strictly on the provided context.

CONTEXT (retrieved from your documents):
{context}

---

INSTRUCTIONS:
- Answer ONLY using information from the context above
- If the context does not contain enough information, say: "I don't have enough information in the provided documents to answer this."
- Be concise but thorough. Use bullet points when listing multiple items.
- Always cite which document/source your information comes from when possible.

QUESTION: {question}

ANSWER:""",
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2E: Build the Full RAG Chain
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_chain(store, llm):
    """
    Assemble the full RAG chain using LangChain Expression Language (LCEL).
    
    Chain Architecture:
        Question
           ↓
        [Retriever] → fetch top-K relevant chunks
           ↓
        [Prompt]    → format context + question
           ↓
        [LLM]       → generate grounded answer
           ↓
        [Parser]    → extract string response
    """
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": int(os.getenv("TOP_K_RESULTS", 4))},
    )

    def format_docs(docs: List[Document]) -> str:
        """Convert retrieved docs into a single numbered context block."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Document")
            page   = doc.metadata.get("page", "")
            header = f"[Source {i}: {source}" + (f", Page {page}" if page else "") + "]"
            formatted.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2F: Query with Sources
# ─────────────────────────────────────────────────────────────────────────────

def query_rag(
    query: str,
    store=None,
    llm=None,
    stream: bool = False,
) -> dict:
    """
    Run a full RAG query and return the answer with source citations.
    
    Returns:
        {
            "answer": "...",
            "sources": [{"content": "...", "source": "...", "page": ...}, ...],
            "query": "..."
        }
    """
    if store is None:
        store = load_vector_store(
            store_type=os.getenv("VECTOR_STORE", "faiss"),
            persist_path=os.getenv("VECTOR_STORE_PATH", "./vectorstore"),
            embed_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
    if llm is None:
        llm = get_llm()

    chain, retriever = build_rag_chain(store, llm)

    # Get sources for citation
    retrieved_docs = retriever.invoke(query)
    sources = [
        {
            "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", None),
        }
        for doc in retrieved_docs
    ]

    if stream:
        return chain.stream(query), sources

    answer = chain.invoke(query)
    return {"answer": answer, "sources": sources, "query": query}


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Query Engine")
    parser.add_argument("--query", required=True, help="Your question")
    parser.add_argument("--store", default=None, help="'faiss' or 'chroma'")
    parser.add_argument("--path",  default=None, help="Path to vector store")
    args = parser.parse_args()

    store_type   = args.store or os.getenv("VECTOR_STORE", "faiss")
    persist_path = args.path  or os.getenv("VECTOR_STORE_PATH", "./vectorstore")

    vectorstore = load_vector_store(store_type, persist_path)
    llm         = get_llm()

    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"{'='*60}\n")

    result = query_rag(args.query, vectorstore, llm)
    print(f"Answer:\n{result['answer']}\n")
    print("="*60)
    print(f"Sources ({len(result['sources'])} chunks retrieved):")
    for i, src in enumerate(result['sources'], 1):
        print(f"\n  [{i}] {src['source']}" + (f" (p.{src['page']})" if src['page'] else ""))
        print(f"      {src['content'][:150]}...")
    print(f"{'='*60}")
