<!-- docs:meta
topic_id: repo.docs.training.finetune.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.training.finetune.readme
-->

# Sounio LoRA Fine-Tuning Pipeline

Fine-tune code LLMs to generate idiomatic Sounio code using LoRA (Low-Rank
Adaptation). The pipeline teaches models Sounio-specific syntax patterns that
differ from Rust and other languages they were trained on.

## Why Fine-Tune?

LLMs trained on general code corpora produce Rust-like syntax when asked to
write Sounio. Common errors include:

- Adding semicolons (`let x = 5;` instead of `let x = 5`)
- Using `&mut` instead of `&!`
- Using `let mut` instead of `var`
- Using Rust macros (`assert!()` instead of `assert()`)
- Missing effects declarations (`with IO, Mut, Panic, Div`)
- Using unary minus (`-42` instead of `0 - 42`)

LoRA fine-tuning on the Sounio corpus corrects these patterns with minimal
compute cost (~2 hours on a single A100).

## Pipeline Overview

```
1. Collect corpus     docs/training/finetune/prepare_corpus.sh
                      -> docs/training/finetune/sounio_corpus.txt

2. Prepare instructions  docs/training/instructions/sounio_instruct.jsonl
                         (instruction-completion pairs, already provided)

3. Validate assets    bash scripts/ci/lora_assets_gate.sh

4. Fine-tune          python docs/training/finetune/lora_finetune.py \
                        --corpus docs/training/finetune/sounio_corpus.txt \
                        --instructions docs/training/instructions/sounio_instruct.jsonl \
                        --output models/sounio-coder-lora

5. Evaluate           python benchmarks/multipl_e/eval_sounio.py \
                        --dir generated/ --k 1,10,100
```

## Quick Start

### 1. Install Dependencies

```bash
pip install transformers>=4.38 peft>=0.9 datasets>=2.17 \
            accelerate>=0.27 bitsandbytes>=0.42 trl>=0.7
```

### 2. Prepare the Corpus

```bash
bash docs/training/finetune/prepare_corpus.sh
```

This collects all `.sio` files from `stdlib/`, `tests/`, `examples/`, and
`benchmarks/` into a single `sounio_corpus.txt`.

By default, the corpus builder excludes `stdlib/ontology/**` so active ontology
work can proceed in the main checkout without being accidentally absorbed into a
training artifact. Set `LORA_CORPUS_INCLUDE_ONTOLOGY=1` only when that lane is
stable and intended for the training corpus.

### 3. Validate Dataset Assets

```bash
bash scripts/ci/lora_assets_gate.sh
```

This is a CPU-only, dependency-light gate. It checks the corpus markers,
instruction dataset provenance, JSONL shape, and contrastive syntax-fix
examples before any model download or GPU job.

### 4. Run Fine-Tuning

```bash
python docs/training/finetune/lora_finetune.py \
    --corpus docs/training/finetune/sounio_corpus.txt \
    --instructions docs/training/instructions/sounio_instruct.jsonl \
    --output models/sounio-coder-lora \
    --epochs 3 \
    --lr 2e-4 \
    --rank 16
```

### 5. Validate Model Setup

```bash
python docs/training/finetune/lora_finetune.py \
    --corpus docs/training/finetune/sounio_corpus.txt \
    --output models/sounio-coder-lora \
    --validate --dry-run
```

## Configuration

### Base Models

| Model | Size | VRAM (4-bit) | Notes |
|-------|------|-------------|-------|
| `bigcode/starcoder2-3b` | 3B | ~6 GB | Default. Good balance of quality and speed. |
| `bigcode/starcoder2-7b` | 7B | ~10 GB | Better quality, slower training. |
| `codellama/CodeLlama-7b-hf` | 7B | ~10 GB | Alternative base. |
| `Qwen/Qwen2.5-Coder-3B` | 3B | ~6 GB | Strong multilingual code model. |

### LoRA Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rank` | 16 | LoRA rank. Higher = more capacity. 8-64 typical. |
| `--alpha` | 32 | LoRA scaling. Usually 2x rank. |
| `--dropout` | 0.05 | LoRA dropout. 0.05-0.1 typical. |
| `--target-modules` | auto | Attention + MLP projections. |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Training epochs. 2-5 typical for small corpora. |
| `--lr` | 2e-4 | Learning rate. 1e-4 to 5e-4 typical for LoRA. |
| `--batch-size` | 4 | Per-device batch size. Reduce if OOM. |
| `--grad-accum` | 4 | Gradient accumulation steps. Effective batch = batch * accum. |
| `--max-seq-len` | 2048 | Maximum token sequence length. |
| `--warmup-ratio` | 0.03 | LR warmup fraction. |

## Data Format

### Corpus (`sounio_corpus.txt`)

Raw `.sio` source files concatenated with file markers:

```
// === FILE: stdlib/core/option.sio ===
struct IntOption { value: i64, is_some: bool }

impl IntOption {
    fn none() -> IntOption { ... }
    fn some(v: i64) -> IntOption { ... }
}

// === FILE: tests/run-pass/hello.sio ===
fn main() with IO {
    println("Hello, Sounio!")
}
```

### Instructions (`sounio_instruct.jsonl`)

JSON Lines format with instruction-completion pairs:

```json
{"instruction": "Write a Sounio hello world program", "completion": "fn main() with IO {\n    println(\"Hello, Sounio!\")\n}"}
{"instruction": "Write a Sounio function that computes factorial", "completion": "fn factorial(n: i64) -> i64 with Mut, Panic, Div {\n    ..."}
```

## Hardware Requirements

| Configuration | VRAM | Training Time (3B, 3 epochs) |
|--------------|------|------------------------------|
| 1x A100 80GB | ~20 GB | ~1 hour |
| 1x A10G 24GB | ~16 GB | ~2 hours |
| 1x RTX 4090 24GB | ~16 GB | ~2 hours |
| 1x RTX 4080 16GB | ~12 GB (4-bit) | ~3 hours |
| 1x RTX 3090 24GB | ~16 GB | ~3 hours |

4-bit quantization is enabled by default and reduces VRAM usage by ~60%
with minimal quality loss.

## Evaluation

After fine-tuning, evaluate using the MultiPL-E benchmark:

```bash
# Generate completions (use your preferred inference setup)
# Then evaluate:
python benchmarks/multipl_e/eval_sounio.py \
    --dir generated/ \
    --k 1,10,100 \
    --output results.jsonl \
    --verbose
```

Key metrics:
- **pass@1**: Fraction of problems solved on first try
- **pass@10**: Fraction of problems solvable with 10 attempts
- **Sounio syntax compliance**: Absence of semicolons, &mut, let mut, Rust macros

## Advanced Usage

### Resume from Checkpoint

```bash
python lora_finetune.py \
    --corpus sounio_corpus.txt \
    --output models/sounio-lora \
    --resume models/sounio-lora/checkpoint-500
```

### Push to HuggingFace Hub

```bash
python lora_finetune.py \
    --corpus sounio_corpus.txt \
    --output models/sounio-lora \
    --push-to-hub sounio/starcoder2-3b-sounio-lora
```

### Weights & Biases Logging

```bash
pip install wandb
python lora_finetune.py \
    --corpus sounio_corpus.txt \
    --output models/sounio-lora \
    --report-to wandb
```

### Using a Different Base Model

```bash
python lora_finetune.py \
    --base-model Qwen/Qwen2.5-Coder-3B \
    --corpus sounio_corpus.txt \
    --output models/sounio-qwen-lora \
    --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```
