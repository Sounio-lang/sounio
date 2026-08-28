---
license: apache-2.0
language:
- en
pretty_name: Sounio AI Selector Distill
task_categories:
- text-generation
tags:
- code
- compiler
- programming-language
- sounio
- compiler-in-the-loop
- distillation
---

# sounio-ai-selector-distill

Compiler-selector distillation records for the v12 Sounio LoRA rung.

This dataset is built from pass@k eval artifacts that are not the fresh v2
promotion gate. For each prompt where one generated candidate passes
`souc check`, the builder records the compiler-selected candidate as the target
output. If earlier candidates failed, they are included as contrastive context.

The purpose is narrow: teach the model to put the compiler-selected candidate
at `sample0`, improving compile@1/scientific@1 without switching to a larger
base model.

## Files

- `selector_distill.v1.jsonl`: instruction/input/output training records.
- `manifest.v1.json`: source eval metadata, selected sample distribution,
  category counts, and skipped prompt counts.

## Rebuild

```bash
python3 scripts/dev/build_sounio_ai_selector_distill_dataset.py \
  --results-jsonl <pass-at-k-results.jsonl> \
  --prompts datasets/sounio-ai-eval/prompts.v1.jsonl \
  --model-key repair_v04_direct \
  --prefer-run-pass
```

For v12, the Slurm wrapper builds this dataset inside the job from a v1
selector pass, then evaluates the trained adapter on the held-out v2 prompt set.
