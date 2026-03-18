#!/usr/bin/env python3
"""
LoRA fine-tuning script for Sounio language support.

Fine-tunes StarCoder2-3B (or a compatible code LLM) on the Sounio corpus and
instruction data to teach the model idiomatic Sounio syntax and patterns.

Requirements:
    pip install transformers>=4.38 peft>=0.9 datasets>=2.17 \
                accelerate>=0.27 bitsandbytes>=0.42 trl>=0.7

Usage:
    # Basic fine-tune on corpus only
    python lora_finetune.py \
        --corpus training/finetune/sounio_corpus.txt \
        --output models/sounio-coder-lora

    # Full fine-tune with instruction data
    python lora_finetune.py \
        --corpus training/finetune/sounio_corpus.txt \
        --instructions training/instructions/sounio_instruct.jsonl \
        --output models/sounio-coder-lora \
        --epochs 3 --lr 2e-4 --rank 16

    # Resume from checkpoint
    python lora_finetune.py \
        --corpus training/finetune/sounio_corpus.txt \
        --output models/sounio-coder-lora \
        --resume models/sounio-coder-lora/checkpoint-500

    # Push to HuggingFace Hub
    python lora_finetune.py \
        --corpus training/finetune/sounio_corpus.txt \
        --output models/sounio-coder-lora \
        --push-to-hub sounio/starcoder2-3b-sounio-lora

Hardware:
    - Minimum: 1x GPU with 16GB VRAM (e.g., RTX 4080, A4000)
    - With 4-bit quantization, fits on 8GB VRAM for inference
    - Training uses gradient checkpointing to reduce memory
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sounio-finetune")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "bigcode/starcoder2-3b"
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 4
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-4
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_WARMUP_RATIO = 0.03

# LoRA target modules for StarCoder2 architecture
# These are the attention and MLP projection layers
STARCODER2_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Instruction template for chat-style fine-tuning
INSTRUCTION_TEMPLATE = """\
### Instruction:
{instruction}

### Response:
{completion}"""

# File header prepended to corpus chunks
CORPUS_HEADER = "// Sounio programming language (.sio)\n"


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    corpus_path: Optional[str] = None
    instructions_path: Optional[str] = None
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    corpus_overlap: int = 256  # token overlap between corpus chunks


def load_corpus(path: str, tokenizer, max_seq_len: int, overlap: int = 256) -> list[dict]:
    """Load and chunk a raw .sio corpus file for causal LM training.

    Splits the corpus into overlapping chunks of max_seq_len tokens.
    Each chunk is prepended with a file-type header so the model learns
    to associate the token patterns with Sounio.

    Args:
        path: Path to the corpus text file.
        tokenizer: HuggingFace tokenizer instance.
        max_seq_len: Maximum sequence length in tokens.
        overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of dicts with "text" key for each chunk.
    """
    logger.info("Loading corpus from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Tokenize the entire corpus
    tokens = tokenizer.encode(raw_text, add_special_tokens=False)
    logger.info("Corpus: %d characters, %d tokens", len(raw_text), len(tokens))

    # Split into overlapping chunks
    chunks = []
    stride = max_seq_len - overlap
    if stride <= 0:
        stride = max_seq_len // 2

    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + max_seq_len]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        # Prepend header to first chunk or chunks that start at a file boundary
        if i == 0 or "// === FILE:" in chunk_text[:200]:
            chunk_text = CORPUS_HEADER + chunk_text
        else:
            chunk_text = "// (continued)\n" + chunk_text

        chunks.append({"text": chunk_text})
        i += stride

    logger.info("Created %d corpus chunks (stride=%d, overlap=%d)", len(chunks), stride, overlap)
    return chunks


def load_instructions(path: str) -> list[dict]:
    """Load instruction-completion pairs from a JSONL file.

    Each line should be a JSON object with "instruction" and "completion" keys.
    The pairs are formatted using INSTRUCTION_TEMPLATE for chat-style training.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dicts with "text" key containing the formatted instruction.
    """
    logger.info("Loading instructions from %s", path)
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: invalid JSON: %s", line_num, e)
                continue

            instruction = obj.get("instruction", "")
            completion = obj.get("completion", "")

            if not instruction or not completion:
                logger.warning("Line %d: missing instruction or completion", line_num)
                continue

            formatted = INSTRUCTION_TEMPLATE.format(
                instruction=instruction,
                completion=completion,
            )
            items.append({"text": formatted})

    logger.info("Loaded %d instruction pairs", len(items))
    return items


def prepare_dataset(config: DatasetConfig, tokenizer):
    """Prepare the combined training dataset.

    Merges corpus chunks and instruction pairs into a single dataset.

    Args:
        config: Dataset configuration.
        tokenizer: HuggingFace tokenizer.

    Returns:
        HuggingFace Dataset ready for training.
    """
    from datasets import Dataset

    all_items = []

    if config.corpus_path and os.path.isfile(config.corpus_path):
        corpus_items = load_corpus(
            config.corpus_path,
            tokenizer,
            config.max_seq_len,
            config.corpus_overlap,
        )
        all_items.extend(corpus_items)
    else:
        logger.warning("No corpus file found at %s", config.corpus_path)

    if config.instructions_path and os.path.isfile(config.instructions_path):
        instruct_items = load_instructions(config.instructions_path)
        all_items.extend(instruct_items)
    else:
        logger.info("No instruction file specified or found")

    if not all_items:
        raise ValueError(
            "No training data found. Provide --corpus and/or --instructions."
        )

    logger.info("Total training examples: %d", len(all_items))
    dataset = Dataset.from_list(all_items)
    return dataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    use_flash_attn: bool = False,
):
    """Load the base model with optional 4-bit quantization.

    Args:
        model_name: HuggingFace model identifier or local path.
        use_4bit: Enable 4-bit quantization via bitsandbytes.
        use_flash_attn: Enable FlashAttention-2 (requires compatible GPU).

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config
    quant_config = None
    if use_4bit:
        logger.info("Using 4-bit quantization (NF4)")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {
        "quantization_config": quant_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    logger.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    logger.info(
        "Model loaded: %s parameters, %.1f GB",
        f"{sum(p.numel() for p in model.parameters()):,}",
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9,
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def setup_lora(
    model,
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: Optional[list[str]] = None,
):
    """Apply LoRA adapters to the model.

    Args:
        model: Base model to adapt.
        rank: LoRA rank (r). Higher = more capacity, more memory.
        alpha: LoRA scaling factor. Common: alpha = 2 * rank.
        dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.

    Returns:
        Model with LoRA adapters applied.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    logger.info(
        "Setting up LoRA: rank=%d, alpha=%d, dropout=%.2f",
        rank, alpha, dropout,
    )

    # Prepare model for k-bit training (freezes base, casts layernorm to fp32)
    model = prepare_model_for_kbit_training(model)

    modules = target_modules or STARCODER2_TARGET_MODULES
    logger.info("Target modules: %s", modules)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Fan-in/fan-out for better initialization
        fan_in_fan_out=False,
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied: %s trainable / %s total (%.2f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        100.0 * trainable_params / total_params,
    )

    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    resume_from: Optional[str] = None,
    report_to: str = "none",
):
    """Run the LoRA fine-tuning training loop.

    Uses the SFTTrainer from trl for supervised fine-tuning on the
    text field of each dataset example.

    Args:
        model: Model with LoRA adapters.
        tokenizer: Tokenizer instance.
        dataset: HuggingFace Dataset with "text" column.
        output_dir: Directory to save checkpoints and final adapter.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        max_seq_len: Maximum sequence length for tokenization.
        warmup_ratio: Fraction of steps for LR warmup.
        resume_from: Path to checkpoint to resume from.
        report_to: Logging integration ("none", "wandb", "tensorboard").
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    effective_batch = batch_size * grad_accum
    total_steps = (len(dataset) * epochs) // effective_batch
    logger.info(
        "Training: %d examples, %d epochs, batch=%d*%d=%d, %d steps, lr=%.1e",
        len(dataset), epochs, batch_size, grad_accum, effective_batch,
        total_steps, lr,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False) and torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        group_by_length=True,
        report_to=report_to,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        packing=True,  # pack short examples together for efficiency
    )

    logger.info("Starting training...")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    logger.info("Training complete. Saving adapter to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config for reproducibility
    config_path = os.path.join(output_dir, "training_config.json")
    config = {
        "base_model": model.config._name_or_path if hasattr(model.config, "_name_or_path") else "unknown",
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "max_seq_len": max_seq_len,
        "warmup_ratio": warmup_ratio,
        "total_examples": len(dataset),
        "total_steps": total_steps,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Training config saved to %s", config_path)

    return trainer


# ---------------------------------------------------------------------------
# Inference / validation
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a completion from the fine-tuned model.

    Args:
        model: Fine-tuned model.
        tokenizer: Tokenizer instance.
        prompt: Input prompt text.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text (prompt + completion).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def validate_model(model, tokenizer) -> None:
    """Run quick validation prompts to check fine-tuning quality.

    Tests that the model generates Sounio-specific patterns:
    - var instead of let mut
    - &! instead of &mut
    - No semicolons
    - Effects declarations
    """
    test_prompts = [
        "// Write a Sounio function that adds two integers\nfn",
        "// Sounio mutable variable\nvar",
        "### Instruction:\nWrite a Sounio hello world program\n\n### Response:\n",
        "fn factorial(n: i64) -> i64",
    ]

    logger.info("Running validation prompts...")
    for i, prompt in enumerate(test_prompts, 1):
        output = run_inference(model, tokenizer, prompt, max_new_tokens=128)
        logger.info("--- Validation %d ---", i)
        logger.info("Prompt: %s", prompt[:80])
        logger.info("Output: %s", output[:200])

        # Check for common Sounio patterns
        has_var = "var " in output
        has_effects = "with " in output
        has_semicolons = ";\n" in output or "; " in output
        has_mut_ref = "&mut" in output

        if has_semicolons:
            logger.warning("  WARNING: Output contains semicolons (not Sounio)")
        if has_mut_ref:
            logger.warning("  WARNING: Output contains &mut (should be &!)")


# ---------------------------------------------------------------------------
# Hub upload
# ---------------------------------------------------------------------------

def push_to_hub(model, tokenizer, repo_id: str) -> None:
    """Push the LoRA adapter to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "sounio/starcoder2-3b-sounio-lora").
    """
    logger.info("Pushing to HuggingFace Hub: %s", repo_id)
    model.push_to_hub(repo_id, private=True)
    tokenizer.push_to_hub(repo_id, private=True)
    logger.info("Successfully pushed to %s", repo_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning script for Sounio language support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Fine-tune on corpus
    python lora_finetune.py --corpus sounio_corpus.txt --output models/sounio-lora

    # Fine-tune with instructions
    python lora_finetune.py --corpus sounio_corpus.txt \\
                            --instructions sounio_instruct.jsonl \\
                            --output models/sounio-lora --epochs 3

    # Use a different base model
    python lora_finetune.py --corpus sounio_corpus.txt \\
                            --base-model codellama/CodeLlama-7b-hf \\
                            --output models/sounio-codellama-lora
        """,
    )

    # Data
    parser.add_argument(
        "--corpus",
        type=str,
        help="Path to Sounio corpus text file (from prepare_corpus.sh)",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        help="Path to instruction JSONL file",
    )

    # Model
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model to fine-tune (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the LoRA adapter",
    )

    # LoRA hyperparameters
    parser.add_argument(
        "--rank", "-r",
        type=int,
        default=DEFAULT_LORA_RANK,
        help=f"LoRA rank (default: {DEFAULT_LORA_RANK})",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
        help=f"LoRA alpha (default: {DEFAULT_LORA_ALPHA})",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_LORA_DROPOUT,
        help=f"LoRA dropout (default: {DEFAULT_LORA_DROPOUT})",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        help="LoRA target modules (default: auto-detect for model architecture)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Per-device batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
        help=f"Gradient accumulation steps (default: {DEFAULT_GRAD_ACCUM})",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"Maximum sequence length (default: {DEFAULT_MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help=f"Warmup ratio (default: {DEFAULT_WARMUP_RATIO})",
    )

    # Options
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (uses more VRAM)",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Enable FlashAttention-2",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume training from checkpoint directory",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        help="Push adapter to HuggingFace Hub (repo ID)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation prompts after training",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=["none", "wandb", "tensorboard"],
        help="Logging integration (default: none)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare dataset and model but skip training",
    )

    args = parser.parse_args()

    if not args.corpus and not args.instructions:
        parser.error("At least one of --corpus or --instructions is required")

    # --- Load model and tokenizer ---
    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        use_4bit=not args.no_4bit,
        use_flash_attn=args.flash_attn,
    )

    # --- Prepare dataset ---
    ds_config = DatasetConfig(
        corpus_path=args.corpus,
        instructions_path=args.instructions,
        max_seq_len=args.max_seq_len,
    )
    dataset = prepare_dataset(ds_config, tokenizer)

    # --- Apply LoRA ---
    model = setup_lora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
    )

    if args.dry_run:
        logger.info("Dry run complete. Dataset: %d examples. Exiting.", len(dataset))
        return

    # --- Train ---
    os.makedirs(args.output, exist_ok=True)
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        warmup_ratio=args.warmup_ratio,
        resume_from=args.resume,
        report_to=args.report_to,
    )

    # --- Validate ---
    if args.validate:
        validate_model(model, tokenizer)

    # --- Push to Hub ---
    if args.push_to_hub:
        push_to_hub(model, tokenizer, args.push_to_hub)

    logger.info("All done. Adapter saved to %s", args.output)


if __name__ == "__main__":
    main()
