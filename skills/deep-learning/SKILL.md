---
name: deep-learning
description: Train, evaluate, and manage deep-learning models and datasets for the Sounio project
user-invocable: true
allowed-tools: Bash, Read, Edit, Write, Glob, Grep
---

# Deep Learning

Use this skill for ML/DL work tied to Sounio:
- Training models on `datasets/sounio-*`
- Fine-tuning code-generation or repair models
- Evaluating on `benchmarks/` or custom splits
- Managing checkpoints, configs, and runs on GPU/Slurm

## Datasets

```bash
ls datasets/
# sounio-ai-eval, sounio-ai-repair, sounio-code-examples, sounio-contrastive, ...
```

Inspect manifests before training:

```bash
head datasets/sounio-code-examples/manifest.json
head datasets/sounio-code-examples/train.jsonl
```

## Training loop pattern

1. Confirm the dataset split and task objective.
2. Check GPU availability: `nvidia-smi` or `sinfo` via Foundry/Slurm.
3. Run training in an isolated venv / conda env. Prefer `/workspace` or Foundry scratch, never the live repo root for large outputs.
4. Log metrics to a run directory under `artifacts/llm_training/` or the designated foundry output path.
5. Evaluate on the held-out split and report exact metrics.

## Lightweight eval

```bash
python -m py_compile scripts/ml/*.py
python scripts/ml/quick_eval.py --dataset datasets/sounio-ai-eval --model <path>
```

## Heavy training

For multi-GPU or long runs, submit through Slurm/BeagleCockpit MCP. Do not run long trainings interactively in `/workspace/sounio`.

See `docs/ops/foundry_slurm_handoff.md`.

## Checkpoints and artifacts

- Store checkpoints outside the main source tree when possible.
- If committing a small result table or config, keep it under `artifacts/llm_training/` or `docs/research/`.
- Never commit large `.pt`, `.safetensors`, or dataset cache files.

## Offload requirements

External-facing artifacts (papers, benchmark reports, IRB materials) require fan-out offload. Math claims inside training analysis require math-review offload.

## Common paths

| Asset | Path |
|---|---|
| Code examples dataset | `datasets/sounio-code-examples/` |
| Repair dataset | `datasets/sounio-ai-repair/` |
| Training scripts | `scripts/ml/`, `bin/llm-offload` |
| Benchmark results | `benchmarks/results/` |
| Checkpoints | `artifacts/llm_training/` |
