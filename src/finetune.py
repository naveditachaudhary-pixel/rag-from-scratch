"""
Fine-Tuning Pipeline — Step 2: LoRA / QLoRA Fine-Tuning
=========================================================
Fine-tunes a small open-source LLM on your Q&A dataset using:
  - PEFT (LoRA adapters -- only ~1% of params trained)
  - TRL SFTTrainer (clean instruction-tuning loop)
  - Optional 4-bit QLoRA via bitsandbytes (requires GPU, halves VRAM)

Supported base models (set via --model or FINETUNE_MODEL in .env):
  TinyLlama/TinyLlama-1.1B-Chat-v1.0   (~600MB download, fastest, good for testing)
  microsoft/Phi-3-mini-4k-instruct      (~2.4GB, excellent quality, recommended)
  meta-llama/Llama-3.2-1B-Instruct     (~1.2GB, needs HuggingFace token)
  meta-llama/Llama-3.2-3B-Instruct     (~2.0GB, needs HuggingFace token)

Hardware requirements:
  CPU only    -> TinyLlama, no quantization, batch_size=1, slow (hours)
  GPU  6GB+   -> Phi-3-mini with 4-bit QLoRA
  GPU 12GB+   -> Phi-3-mini full LoRA, or Llama-3.2-3B with QLoRA
  GPU 24GB+   -> Any model without quantization

Usage:
  python src/finetune.py
  python src/finetune.py --model microsoft/Phi-3-mini-4k-instruct --epochs 3
  python src/finetune.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quantize
  python src/finetune.py --dataset ./data/training/qa_pairs.jsonl --epochs 5
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Import dependencies with helpful errors
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies(use_4bit: bool = False):
    missing = []
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import peft
    except ImportError:
        missing.append("peft")
    try:
        import trl
    except ImportError:
        missing.append("trl")
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    if use_4bit:
        try:
            import bitsandbytes
        except ImportError:
            missing.append("bitsandbytes")
    if missing:
        raise ImportError(
            f"\nMissing packages: {', '.join(missing)}\n"
            f"Install with:\n  pip install {' '.join(missing)}\n"
            f"Or run:  pip install -r requirements-finetune.txt"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Format Dataset
# ─────────────────────────────────────────────────────────────────────────────

ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def load_jsonl_dataset(path: str):
    """Load a JSONL file into a HuggingFace Dataset."""
    from datasets import Dataset

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} training examples from {path}")
    return Dataset.from_list(records)


def format_alpaca(example: dict) -> dict:
    """Format a single example into the Alpaca prompt format."""
    text = ALPACA_PROMPT.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", ""),
    )
    return {"text": text}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Load Base Model + Apply LoRA
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = False,
    hf_token: Optional[str] = None,
):
    """
    Load the base model and tokenizer.
    Optionally applies 4-bit NF4 quantization (QLoRA).

    WHY LoRA?
    Instead of updating all parameters (billions of weights), LoRA injects
    small trainable matrices (rank r) into the attention layers. This reduces:
      - Trainable params: from ~1B to ~1-8M  (>99% frozen)
      - GPU VRAM: from ~14GB to ~4-6GB
      - Training time: from days to hours
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
    kwargs = {"token": token} if token else {}

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)

    # Ensure padding token exists (many models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4-bit Quantization config (QLoRA)
    bnb_config = None
    if use_4bit:
        logger.info("Enabling 4-bit QLoRA quantization (bitsandbytes)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for LLMs
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,     # nested quantization for extra savings
        )

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        torch_dtype=torch.float16 if not use_4bit else None,
        **kwargs,
    )

    if use_4bit:
        # Required for QLoRA: enable gradient checkpointing
        model.gradient_checkpointing_enable()
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    logger.info(f"Model loaded | Parameters: {model.num_parameters():,}")
    return model, tokenizer


def apply_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
):
    """
    Inject LoRA adapters into the model's attention layers.

    The adapter matrices A (r x d) and B (d x r) are initialized so that
    BA = 0 at training start — meaning LoRA starts as a no-op and the model
    behaves identically at step 0. This ensures stable training from a
    pretrained checkpoint.

    r (rank): higher rank = more parameters = more capacity but slower training
      Typical values: 8 (fast) to 64 (high capacity)
      Recommendation: 16 for most use cases
    """
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        # These projections cover the full attention head in most architectures
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Train with SFTTrainer
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
    warmup_ratio: float = 0.05,
    logging_steps: int = 10,
    save_steps: int = 50,
):
    """
    Run the supervised fine-tuning loop with TRL's SFTTrainer.

    SFTTrainer handles:
    - Tokenization and prompt masking (model only learns from response, not instruction)
    - Example packing (bins multiple short examples into one sequence for efficiency)
    - Gradient accumulation (simulates large batches on small GPU)
    - Checkpoint saving and resumption
    """
    from trl import SFTTrainer, SFTConfig
    import torch

    # Train/val split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    logger.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),          # use FP16 if GPU available
        use_cpu=not torch.cuda.is_available(),   # use CPU if GPU not available
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=True,                            # pack short examples for speed
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps",                   # renamed from evaluation_strategy in TRL 0.28+
        eval_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",                        # disable wandb/tensorboard
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )

    logger.info("Starting fine-tuning...")
    logger.info(f"  Effective batch size: {batch_size * grad_accum}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Learning rate: {learning_rate}")

    # Check for existing checkpoints to resume from
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    logger.info(f"Training complete! Saving LoRA adapter to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Adapter saved.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Test the Fine-Tuned Model
# ─────────────────────────────────────────────────────────────────────────────

def test_inference(adapter_dir: str, test_question: str = "What is RAG?"):
    """
    Load the saved LoRA adapter and run a quick inference test.
    The base model is loaded fresh and the adapter is merged on top.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    # Read base model name from adapter config
    import json
    adapter_config_path = Path(adapter_dir) / "adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]

    logger.info(f"Loading fine-tuned model for inference test...")
    logger.info(f"  Base model : {base_model_name}")
    logger.info(f"  Adapter    : {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()  # merge adapter into base weights for fastest inference

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt = ALPACA_PROMPT.format(
        instruction=test_question,
        input="",
        output="",  # leave blank — model fills this in
    )

    logger.info(f"\nTest question: {test_question}")
    result = pipe(prompt, max_new_tokens=256, temperature=0.1, do_sample=True)
    generated = result[0]["generated_text"]

    # Extract only the response part
    response_start = generated.find("### Response:\n") + len("### Response:\n")
    response = generated[response_start:].strip()

    print(f"\n{'='*60}")
    print(f"Question : {test_question}")
    print(f"{'-'*60}")
    print(f"Answer   : {response}")
    print(f"{'='*60}")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: (Optional) Export to GGUF for Ollama
# ─────────────────────────────────────────────────────────────────────────────

def export_to_ollama_instructions(adapter_dir: str, model_name: str):
    """
    Print instructions for exporting the fine-tuned model to Ollama
    so it can be used as a drop-in replacement in the RAG system.
    """
    modelfile = f"""FROM {adapter_dir}

SYSTEM You are a helpful AI assistant fine-tuned on domain-specific Q&A data.
Always answer based on facts, and say "I don't know" when uncertain.

PARAMETER temperature 0.1
PARAMETER top_p 0.9
"""
    modelfile_path = Path(adapter_dir) / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile)

    print(f"""
{'='*60}
To use your fine-tuned model with Ollama:

1. Install Ollama: https://ollama.ai
2. Run these commands:

   ollama create {model_name}-finetuned -f {modelfile_path}
   ollama run {model_name}-finetuned

3. Update your .env:

   LLM_PROVIDER=ollama
   LLM_MODEL={model_name}-finetuned

4. Restart the RAG server:

   python src/app.py

{'='*60}
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA/QLoRA Fine-Tuning Pipeline")
    parser.add_argument("--model",       default=os.getenv("FINETUNE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                        help="HuggingFace model ID to fine-tune")
    parser.add_argument("--dataset",     default="./data/training/sample_qa.jsonl",
                        help="Path to JSONL training dataset")
    parser.add_argument("--output",      default="./models/lora-adapter",
                        help="Directory to save the LoRA adapter")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--lora-r",      type=int,   default=32,  help="LoRA rank")
    parser.add_argument("--quantize",    action="store_true",     help="Enable 4-bit QLoRA")
    parser.add_argument("--hf-token",    default=None,            help="HuggingFace token (for gated models)")
    parser.add_argument("--test-only",   action="store_true",     help="Skip training, just test saved adapter")
    parser.add_argument("--test-q",      default="What is RAG and how does it work?",
                        help="Question to use for the inference test after training")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  LoRA Fine-Tuning Pipeline")
    print(f"{'='*60}")
    print(f"  Model     : {args.model}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Output    : {args.output}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  LoRA rank : {args.lora_r}")
    print(f"  4-bit     : {args.quantize}")
    print(f"{'='*60}\n")

    # Check deps
    check_dependencies(use_4bit=args.quantize)

    if args.test_only:
        test_inference(args.output, args.test_q)
        return

    # Full training pipeline
    dataset  = load_jsonl_dataset(args.dataset)
    dataset  = dataset.map(format_alpaca)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        use_4bit=args.quantize,
        hf_token=args.hf_token,
    )
    model = apply_lora(model, r=args.lora_r, lora_alpha=args.lora_r * 2)

    train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Test inference
    test_inference(args.output, args.test_q)

    # Print Ollama export instructions
    model_short = args.model.split("/")[-1].lower()
    export_to_ollama_instructions(args.output, model_short)


if __name__ == "__main__":
    main()
