#!/usr/bin/env python3
"""Render a Sounio AI eval summary JSON as a public Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_LABELS = {
    "base": "Qwen2.5-Coder-1.5B base",
    "lora": "Sounio LoRA",
    "smoke": "Reference smoke",
}

SCIENTIFIC_CATEGORIES = {"stdlib_science", "ode_pbpk", "epistemic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="*.summary.json from sounio_ai_eval.py")
    parser.add_argument("--output", type=Path, help="Write Markdown to this file instead of stdout.")
    return parser.parse_args()


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def scientific_rate(model_summary: dict, field: str = "check_pass") -> float | None:
    cats = model_summary.get("categories", {})
    total = 0
    passed = 0
    for category in SCIENTIFIC_CATEGORIES:
        row = cats.get(category)
        if not row:
            continue
        total += int(row.get("total", 0))
        passed += int(row.get(field, 0))
    if total == 0:
        return None
    return passed / total


def model_rows(summary: dict) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for key, value in summary.items():
        if key == "base_vs_lora_delta":
            continue
        if not isinstance(value, dict):
            continue
        if "compile_rate" not in value:
            continue
        rows.append((key, value))
    order = {"base": 0, "lora": 1, "smoke": 2}
    return sorted(rows, key=lambda item: (order.get(item[0], 99), item[0]))


def render(summary: dict) -> str:
    samples_per_prompt = int(summary.get("samples_per_prompt") or 1)
    if samples_per_prompt > 1:
        metric_header = (
            "| Model | Compile@1 | Compile@3 | Compile@5 | Runtime selector | "
            "Raw syntax@1 | Candidate syntax@1 | Scientific@1 | Scientific@5 | Output match |"
        )
        metric_sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    else:
        metric_header = (
            "| Model | Compile rate | Runtime pass | Raw syntax purity | Candidate syntax purity | "
            "Hallucinated API clean | Scientific tasks | Output match |"
        )
        metric_sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Sounio AI Eval Results",
        "",
        f"- Run id: `{summary.get('run_id', 'unknown')}`",
        f"- Prompt count: `{summary.get('prompt_count', 'unknown')}`",
        f"- Samples per prompt: `{samples_per_prompt}`",
        f"- Compiler: `{summary.get('souc', 'unknown')}`",
        f"- Results JSONL: `{summary.get('results_path', 'unknown')}`",
        "",
        metric_header,
        metric_sep,
    ]
    for key, row in model_rows(summary):
        label = DEFAULT_MODEL_LABELS.get(key, key)
        if samples_per_prompt > 1:
            values = [
                label,
                pct(row.get("compile_at_1")),
                pct(row.get("compile_at_3")),
                pct(row.get("compile_at_5")),
                pct(row.get("runtime_pass_rate")),
                pct(row.get("syntax_purity_rate")),
                pct(row.get("candidate_syntax_purity_rate")),
                pct(scientific_rate(row, "check_pass")),
                pct(scientific_rate(row, "check_pass_at_5")),
                pct(row.get("output_match_rate")),
            ]
        else:
            values = [
                label,
                pct(row.get("compile_rate")),
                pct(row.get("runtime_pass_rate")),
                pct(row.get("syntax_purity_rate")),
                pct(row.get("candidate_syntax_purity_rate")),
                pct(row.get("hallucinated_api_clean_rate")),
                pct(scientific_rate(row)),
                pct(row.get("output_match_rate")),
            ]
        lines.append("| " + " | ".join(values) + " |")

    if samples_per_prompt > 1:
        lines.extend(
            [
                "",
                "## Compiler Selector",
                "",
                "| Model | Selected candidates | First passing sample counts |",
                "|---|---:|---|",
            ]
        )
        for key, row in model_rows(summary):
            label = DEFAULT_MODEL_LABELS.get(key, key)
            counts = row.get("first_passing_sample_counts") or {}
            rendered_counts = ", ".join(
                f"`{sample}`: {count}" for sample, count in sorted(counts.items(), key=lambda item: int(item[0]))
            )
            lines.append(
                f"| {label} | {int(row.get('selector_compile_count', 0))} | {rendered_counts or 'none'} |"
            )

    delta = summary.get("base_vs_lora_delta")
    if isinstance(delta, dict):
        lines.extend(
            [
                "",
                "## Base-vs-LoRA Delta",
                "",
                "| Metric | LoRA - base |",
                "|---|---:|",
            ]
        )
        for metric in [
            "compile_rate",
            "runtime_pass_rate",
            "syntax_purity_rate",
            "candidate_syntax_purity_rate",
        ]:
            if metric in delta:
                lines.append(f"| `{metric}` | {float(delta[metric]) * 100:+.1f} pp |")

    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This table reports compiler and harness evidence, not scientific correctness.",
            "- Reference-smoke rows validate the harness only; they are not model-quality evidence.",
            "- Publish base-vs-LoRA claims only from a real generation run with both model keys present.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    markdown = render(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
