---
license: apache-2.0
language:
- en
pretty_name: Sounio AI Science Repair
task_categories:
- text-generation
tags:
- code
- compiler
- programming-language
- scientific-computing
- ode
- pbpk
- ossm
- octonion
- epistemic-computing
---

# sounio-ai-science-repair

Compiler-checked science-structure records for the v10/v11 Sounio LoRA rungs.

This dataset is intentionally narrower than the generic syntax-repair corpus.
It teaches the shape of scientific Sounio programs:

- ODE/RK4 and PBPK programs;
- stdlib science examples;
- `Knowledge` / provenance / epistemic examples;
- O-SSM and conversational state-space examples;
- octonion, Cayley-Dickson, Clifford, and algebra invariant witnesses.

Current `science_repair.v1.jsonl` was generated with compiler-in-the-loop
acceptance and contains 390 records:

- `ode_rk4`: 32
- `pbpk`: 54
- `ossm`: 105
- `octonion_algebra`: 135
- `epistemic_knowledge`: 60
- `stdlib_science`: 4

## Files

- `science_repair.v1.jsonl`: instruction/input/output records.
- `manifest.v1.json`: source globs, exclusion metadata, category counts, and
  compiler-check rejection sample.

## Rebuild

```bash
python3 scripts/dev/build_sounio_ai_science_repair_dataset.py
```

By default the builder excludes sources used by:

- `datasets/sounio-ai-eval/prompts.v1.jsonl`
- `datasets/sounio-ai-eval/prompts.v2.jsonl`

That keeps the v2 full-eval gate held out.

## Eval Slices

The v10/v11 gates also use focused eval slices derived from v2:

```bash
python3 scripts/dev/build_sounio_ai_science_eval_slices.py
```

This writes:

- `prompts.v2.scientific.jsonl`
- `prompts.v2.ode_pbpk.jsonl`
- `prompts.v2.ossm.jsonl`
- `prompts.v2.algebra_octonion.jsonl`
- `prompts.v2.science-slices.manifest.json`

These slices are evaluation surfaces, not training data.

## Training Use

The v11 gate uses `scripts/dev/lora_finetune.py --dataset-mode science_balanced`
to sample all available held-out-safe science-structure records first, then fill
the remainder with syntax-repair guard records. This keeps the rung small and
continues from `repair_v04_direct`; it does not switch to a larger base model.
