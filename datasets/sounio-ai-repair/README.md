# Sounio AI Repair Dataset

`repair.v1.jsonl` is a small BAD-to-GOOD corpus for the next Sounio LoRA loop.
It targets the failure modes observed in the first compiler-in-the-loop eval:
Markdown wrapping, Rust leakage, semicolon termination, `let mut`/`&mut`, Rust
macros, `Vec<T>`, and hallucinated APIs such as `.len()`, `.push()`, `.parse()`,
and `read_line()`.

Generate it from the current eval prompt set and, when available, an eval JSONL:

```bash
python3 scripts/dev/build_sounio_ai_repair_dataset.py \
  --results-jsonl artifacts/sounio-ai-eval/results/<run-id>.jsonl \
  --max-records 240
```

The output keeps both generic fine-tuning fields (`instruction`, `input`,
`output`) and explicit repair-audit fields (`bad`, `good`, `repair_kind`,
`source_prompt_id`, `source_path`, `from_eval_failure`).

This dataset is a syntax discipline rung. It should improve compile rate and
syntax purity before larger scientific-generation claims are made.

`repair.v2-heldout.jsonl` is the held-out-safe companion corpus for the v0.3
runtime gate. It is synthesized from repo `.sio` sources outside
`datasets/sounio-ai-eval/prompts.v1.jsonl`, excluding both prompt ids and
`source_path` values from the evaluation set:

```bash
python3 scripts/dev/build_sounio_ai_repair_dataset.py \
  --exclude-prompts datasets/sounio-ai-eval/prompts.v1.jsonl \
  --source-glob 'tests/run-pass/*.sio' \
  --output datasets/sounio-ai-repair/repair.v2-heldout.jsonl \
  --manifest datasets/sounio-ai-repair/manifest.v2-heldout.json \
  --max-records 1680
```

This allows the v0.3 gate to train on repair patterns without seeing the exact
programs used in its 100-prompt evaluation.
