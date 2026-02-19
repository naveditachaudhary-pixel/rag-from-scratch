"""
Fine-Tuning Pipeline — Step 1: Dataset Preparation
====================================================
Automatically generates Q&A training pairs from your ingested documents.

Two modes:
  A) Auto-generate: Use an LLM to create Q&A pairs from your document chunks
  B) Manual:        Provide your own JSONL file in data/training/

Output format (Alpaca-style JSONL):
  {"instruction": "...", "input": "", "output": "..."}

Usage:
  python src/prepare_dataset.py --source ./data/sample_docs --output ./data/training/qa_pairs.jsonl
  python src/prepare_dataset.py --source ./data/sample_docs --output ./data/training/qa_pairs.jsonl --num-questions 5
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

sys.path.insert(0, os.path.dirname(__file__))
from ingest import load_documents, split_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt to auto-generate Q&A pairs from a text chunk
# ─────────────────────────────────────────────────────────────────────────────

QA_GENERATION_PROMPT = """You are a dataset creation assistant. Given the text below, generate {n} high-quality question-answer pairs suitable for fine-tuning a language model.

Rules:
- Questions should be specific, meaningful, and diverse in style
- Answers must be grounded entirely in the provided text
- Do NOT invent information not present in the text
- Format each pair as a JSON object on its own line: {{"question": "...", "answer": "..."}}
- Output ONLY the JSON lines, no other text

TEXT:
{chunk}

Generate {n} question-answer pairs:"""


def generate_qa_from_chunk(chunk_text: str, llm, num_questions: int = 3) -> List[dict]:
    """Use the LLM to generate Q&A pairs from a document chunk."""
    prompt = QA_GENERATION_PROMPT.format(chunk=chunk_text[:1500], n=num_questions)

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        pairs = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    pair = json.loads(line)
                    if "question" in pair and "answer" in pair:
                        pairs.append({
                            "instruction": pair["question"].strip(),
                            "input": "",
                            "output": pair["answer"].strip(),
                        })
                except json.JSONDecodeError:
                    continue
        return pairs

    except Exception as e:
        logger.warning(f"Failed to generate Q&A for chunk: {e}")
        return []


def prepare_dataset(
    source: str,
    output_path: str,
    num_questions_per_chunk: int = 3,
    max_chunks: int = 50,
) -> int:
    """
    Full dataset preparation pipeline.

    Args:
        source:                   Path to document directory or file
        output_path:              Where to save the JSONL training file
        num_questions_per_chunk:  How many Q&A pairs to generate per chunk
        max_chunks:               Cap on chunks to process (cost/time control)

    Returns:
        Total number of Q&A pairs generated
    """
    from retriever import get_llm

    logger.info(f"Loading documents from: {source}")
    docs   = load_documents(source)
    chunks = split_documents(docs, chunk_size=600, chunk_overlap=50)

    # Cap the number of chunks to avoid excessive LLM calls
    if len(chunks) > max_chunks:
        logger.info(f"Capping at {max_chunks} chunks (total: {len(chunks)})")
        chunks = chunks[:max_chunks]

    logger.info(f"Generating Q&A pairs from {len(chunks)} chunks...")
    llm = get_llm()

    all_pairs = []
    for i, chunk in enumerate(chunks):
        logger.info(f"  [{i+1}/{len(chunks)}] Generating {num_questions_per_chunk} Q&A pairs...")
        pairs = generate_qa_from_chunk(chunk.page_content, llm, num_questions_per_chunk)
        all_pairs.extend(pairs)
        logger.info(f"    → {len(pairs)} pairs generated (total so far: {len(all_pairs)})")

    # Write to JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"\nDataset saved: {output_path}")
    logger.info(f"Total Q&A pairs: {len(all_pairs)}")
    return len(all_pairs)


def add_manual_examples(output_path: str, examples: List[dict]) -> None:
    """
    Append manually written Q&A pairs to the dataset.

    Each example: {"instruction": "question", "input": "", "output": "answer"}
    """
    with open(output_path, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Added {len(examples)} manual examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q&A Dataset Generator for Fine-Tuning")
    parser.add_argument("--source",        default="./data/sample_docs", help="Document source path")
    parser.add_argument("--output",        default="./data/training/qa_pairs.jsonl", help="Output JSONL path")
    parser.add_argument("--num-questions", type=int, default=3,  help="Q&A pairs per chunk")
    parser.add_argument("--max-chunks",    type=int, default=50, help="Max chunks to process")
    args = parser.parse_args()

    total = prepare_dataset(args.source, args.output, args.num_questions, args.max_chunks)
    print(f"\n{'='*50}")
    print(f"Dataset ready: {args.output}")
    print(f"Total pairs  : {total}")
    print(f"{'='*50}")
