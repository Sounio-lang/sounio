#!/usr/bin/env python3
"""
LoRA fine-tune for Sounio syntax knowledge.

Model: DeepSeek-Coder-6.7B-Instruct (or CodeLlama-7B-Instruct)
Method: PEFT LoRA on q_proj + v_proj, rank=16
Dataset: all three datasets merged + stratified

Usage:
    pip install transformers peft datasets torch accelerate bitsandbytes
    python3 scripts/dev/lora_finetune.py [--model deepseek-ai/deepseek-coder-6.7b-instruct]
                                          [--output ./lora-sounio]
                                          [--epochs 3]
                                          [--batch-size 4]
                                          [--dry-run]

Dry-run (no GPU needed):
    python3 scripts/dev/lora_finetune.py --dry-run
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = [
    REPO_ROOT / "datasets/sounio-contrastive/contrastive.jsonl",
    REPO_ROOT / "datasets/sounio-contrastive/synthetic.jsonl",
    REPO_ROOT / "datasets/sounio-code-examples/train.jsonl",
]

SYSTEM_PROMPT = """You are a Sounio programming assistant. Sounio is a systems language that is NOT Rust.

Critical rules:
- No semicolons (statements end at newline)
- `var` for mutable bindings (not `let mut`)
- `&!T` for exclusive/mutable references (not `&mut T`)
- No macros: use `assert()` not `assert!()`, `println()` not `println!()`
- Negative numbers: write `0 - 42` not `-42`
- Functions must declare all effects: `fn f() with IO { ... }`
- No generics except `Knowledge<T>` — use fixed-size arrays `[T; N]`
- No closures — use named function references

Write only valid Sounio code."""


MAX_CHARS = 6000  # ~1500 tokens — filter out files that are too long


def load_dataset() -> list[dict]:
    """Load and merge all datasets into a unified format."""
    records = []

    for path in DATASETS:
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Normalize to {instruction, input, output} format
                if "completion" in r:
                    inp = r.get("input", "")
                    out = r.get("completion", "")
                    # Skip pairs where source or target is too long
                    if len(inp) + len(out) > MAX_CHARS:
                        continue
                    records.append({
                        "instruction": r.get("instruction", "Write correct Sounio code:"),
                        "input": inp,
                        "output": out,
                        "rule": r.get("rule", ""),
                        "source_path": r.get("source_path", ""),
                    })
                elif "instruction" in r and "output" in r:
                    if len(r.get("output", "")) <= MAX_CHARS:
                        records.append(r)

    print(f"Loaded {len(records)} total records")
    return records


def stratified_sample(records: list[dict], target: int = 2000) -> list[dict]:
    """
    Stratify: 40% contrastive (has rule), 30% synthetic (has source_path),
    20% full programs (long completion), 10% effects/types (contains 'with ').
    """
    contrastive = [r for r in records if r.get("rule") and not r.get("source_path")]
    synthetic = [r for r in records if r.get("source_path")]
    full_programs = [r for r in records if len(r.get("output", "")) > 300]
    effects = [r for r in records if "with " in r.get("output", "")]

    def sample(lst, n):
        if len(lst) <= n:
            return lst[:]
        return random.sample(lst, n)

    sampled = (
        sample(contrastive, int(target * 0.40)) +
        sample(synthetic, int(target * 0.30)) +
        sample(full_programs, int(target * 0.20)) +
        sample(effects, int(target * 0.10))
    )
    random.shuffle(sampled)
    return sampled


def format_alpaca(r: dict) -> str:
    """Format a record as Alpaca-style instruction prompt."""
    if r.get("input", "").strip():
        return (
            f"### System\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction\n{r['instruction']}\n\n"
            f"### Input\n{r['input']}\n\n"
            f"### Response\n{r['output']}"
        )
    return (
        f"### System\n{SYSTEM_PROMPT}\n\n"
        f"### Instruction\n{r['instruction']}\n\n"
        f"### Response\n{r['output']}"
    )


def run_dry(records: list[dict]):
    """Validate dataset without GPU."""
    print("\n--- Dry run: dataset validation ---")
    sampled = stratified_sample(records)
    print(f"Stratified sample: {len(sampled)} records")

    # Show distribution
    rules = [r.get("rule", "full_program") for r in sampled]
    from collections import Counter
    for rule, count in Counter(rules).most_common(10):
        print(f"  {rule or 'full_program'}: {count}")

    # Show one formatted example
    print("\n--- Example training record ---")
    ex = sampled[0]
    formatted = format_alpaca(ex)
    print(formatted[:800])
    print("...\n")

    # Token length estimate (rough: 4 chars per token)
    lengths = [len(format_alpaca(r)) // 4 for r in sampled]
    avg = sum(lengths) / len(lengths)
    max_len = max(lengths)
    print(f"Token length estimate: avg={avg:.0f}, max={max_len}")
    if max_len > 2048:
        print(f"  WARNING: {sum(1 for l in lengths if l > 2048)} records exceed 2048 tokens")

    print("\nDry run complete. Run without --dry-run on a GPU machine to train.")


def run_train(records: list[dict], model_name: str, output_dir: str, epochs: int, batch_size: int):
    """Run LoRA fine-tuning."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install transformers peft datasets torch accelerate bitsandbytes")
        sys.exit(1)

    sampled = stratified_sample(records)
    print(f"Training on {len(sampled)} records")

    # Tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format dataset
    texts = [format_alpaca(r) for r in sampled]
    split = int(len(texts) * 0.9)
    train_texts, val_texts = texts[:split], texts[split:]

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

    train_ds = Dataset.from_dict({"text": train_texts}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts}).map(tokenize, batched=True)
    train_ds = train_ds.map(lambda x: {"labels": x["input_ids"]})
    val_ds = val_ds.map(lambda x: {"labels": x["input_ids"]})

    # Model + LoRA
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,  # QLoRA — requires bitsandbytes
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Training...")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved to {output_dir}")
    print("Load with: model = PeftModel.from_pretrained(base_model, '{output_dir}')")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune for Sounio syntax")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-6.7b-instruct",
                        help="Base model (HuggingFace model ID)")
    parser.add_argument("--output", default="./lora-sounio",
                        help="Output directory for LoRA weights")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate dataset without GPU or model download")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading datasets...")
    records = load_dataset()
    if not records:
        print("No records loaded. Check dataset paths.")
        sys.exit(1)

    if args.dry_run:
        run_dry(records)
    else:
        run_train(records, args.model, args.output, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
