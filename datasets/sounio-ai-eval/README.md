---
license: apache-2.0
language:
- en
pretty_name: Sounio AI Eval Prompts
task_categories:
- text-generation
tags:
- code
- compiler
- programming-language
- scientific-computing
- uncertainty-propagation
- evaluation
---

# sounio-ai-eval

Canonical small prompt set for measuring Sounio code-model assistance with the
compiler in the loop.

This dataset is not a training corpus. It is an evaluation surface for comparing
a base code model against a Sounio-specialized adapter under the same prompts and
the same `souc` compiler judge.

## Files

- `prompts.v1.jsonl`: 100 prompt records.
- `manifest.v1.json`: prompt-set metadata and target category mix.
- `prompts.v2.jsonl`: 150 fresh held-out prompt records, source-disjoint from
  v1.
- `manifest.v2.json`: v2 prompt-set metadata, category counts, discovery roots,
  and exclusion metadata.

Each prompt record contains:

- `id`: stable prompt id.
- `category`: one of `basic_syntax`, `functions_control`, `stdlib_science`,
  `numeric_io`, `ode_pbpk`, or `epistemic`.
- `prompt`: natural-language instruction plus the Sounio syntax contract.
- `source_path`: repository reference source for harness smoke tests only.
- `expected_stdout`: optional substring expected in runtime output.
- `run`: whether the harness should attempt `souc run` after `souc check`.
- `difficulty`: coarse source-size-derived difficulty label.
- `tags`: search and reporting tags.

## Rebuild

```bash
python3 scripts/dev/build_sounio_ai_eval_prompts.py
```

Build the fresh v2 set from the local seed dataset plus tracked repository
`.sio` sources, while excluding all v1 source paths:

```bash
python3 scripts/dev/build_sounio_ai_eval_prompts.py \
  --output datasets/sounio-ai-eval/prompts.v2.jsonl \
  --manifest datasets/sounio-ai-eval/manifest.v2.json \
  --name sounio-ai-eval-prompts-v2 \
  --target-size 150 \
  --exclude-prompts datasets/sounio-ai-eval/prompts.v1.jsonl \
  --discover-repo-sources \
  --timeout 20
```

The current v2 set has no `source_path` overlap with v1. Its accepted category
mix is skewed toward `basic_syntax` because the builder only accepts sources
that pass `souc check`, then backfills from validated tracked sources when the
target category mix is not available.

## Evaluate

Use a generator command that accepts the prompt on stdin and prints one complete
Sounio source file on stdout:

```bash
python3 scripts/dev/sounio_ai_eval.py \
  --generator-command 'python3 scripts/dev/local_generate_sounio.py --model {model_id}' \
  --run
```

For a local judge smoke test that does not call any model:

```bash
python3 scripts/dev/sounio_ai_eval.py --use-reference-completions --run --limit 5
```

The harness writes completions and JSONL results under
`artifacts/sounio-ai-eval/`.

The local generator backend is optional and dependency-bearing:

```bash
python3 scripts/dev/local_generate_sounio.py \
  --model Qwen/Qwen2.5-Coder-1.5B
```

For the Sounio LoRA id, the same script loads
`Qwen/Qwen2.5-Coder-1.5B` as the default base and applies the adapter with
PEFT. This matches the adapter's `base_model_name_or_path`.

For GPU evaluation, prefer the batch generator so each model is loaded once:

```bash
python3 scripts/dev/batch_generate_sounio.py \
  --model base=Qwen/Qwen2.5-Coder-1.5B \
  --model lora=chiuratto-AIgourakis/sounio-qwen25-coder-1p5b-lora \
  --prompt-style instruction \
  --max-new-tokens 192 \
  --samples-per-prompt 1 \
  --output-dir artifacts/sounio-ai-eval/raw-generated

python3 scripts/dev/sounio_ai_eval.py \
  --model base=Qwen/Qwen2.5-Coder-1.5B \
  --model lora=chiuratto-AIgourakis/sounio-qwen25-coder-1p5b-lora \
  --completions-dir artifacts/sounio-ai-eval/raw-generated \
  --run
```

Render a public Markdown table from a completed run:

```bash
python3 scripts/dev/sounio_ai_eval_report.py \
  artifacts/sounio-ai-eval/results/<run-id>.summary.json
```

For pass@5 evaluation, generate five samples per prompt and evaluate the
completion directory:

```bash
python3 scripts/dev/batch_generate_sounio.py \
  --model repair=path/to/adapter \
  --prompts datasets/sounio-ai-eval/prompts.v2.jsonl \
  --prompt-style source_only \
  --samples-per-prompt 5 \
  --output-dir artifacts/sounio-ai-eval/raw-generated-v2

python3 scripts/dev/sounio_ai_eval.py \
  --model repair=path/to/adapter \
  --prompts datasets/sounio-ai-eval/prompts.v2.jsonl \
  --completions-dir artifacts/sounio-ai-eval/raw-generated-v2 \
  --samples-per-prompt 5 \
  --run
```
