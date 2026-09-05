---
license: apache-2.0
language:
- en
pretty_name: Sounio Code Examples
size_categories:
- n<1K
task_categories:
- text-generation
tags:
- code
- compiler
- programming-language
- scientific-computing
- formal-verification
- uncertainty-propagation
- algebraic-effects
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
---

# sounio-code-examples

Instruction/completion dataset for **Sounio**, a self-hosted systems + scientific programming language for epistemic computing.

## Contents

- `train.jsonl`: 2327 examples
- `validation.jsonl`: 259 examples
- Total: 2586 examples extracted from `tests/run-pass` and `tests/compile-fail`

Each record contains:

- `instruction`: natural-language prompt derived from test annotations, descriptions, and file names
- `completion`: the full `.sio` source file
- `suite`: `run-pass` or `compile-fail`
- `source_path`: original repository path
- `annotations`: extracted `//@ ...` metadata
- `ignore`: whether the upstream suite currently ignores the example

## Why this dataset exists

This dataset is designed to make Sounio legible to code models quickly:

- run-pass examples teach valid syntax and idioms
- compile-fail examples teach effect discipline, refinement failures, and epistemic boundary checks
- source paths preserve provenance back to the repository test corpus

## Rebuild locally

```bash
python3 scripts/dev/export_hf_dataset.py
```

## Upload

```bash
python3 scripts/dev/export_hf_dataset.py --upload
```

Set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` before uploading.
