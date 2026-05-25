#!/usr/bin/env python3
"""Summarize Sounio AI eval failures into an actionable repair taxonomy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Any


SCIENTIFIC_CATEGORIES = {"stdlib_science", "ode_pbpk", "epistemic"}
RUST_KEYS = {
    "rust_macro",
    "rust_mut_ref",
    "rust_let_mut",
    "rust_vec_type",
    "rust_string_type",
}
FOREIGN_LANG_KEYS = RUST_KEYS | {"python_def", "python_import", "c_include"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Results JSONL from sounio_ai_eval.py")
    parser.add_argument("--output", type=Path, help="Write Markdown to this file.")
    parser.add_argument("--source-label", help="Display label for the JSONL source path.")
    parser.add_argument("--title", default="Sounio AI Eval Failure Taxonomy")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.1f}%"


def count_hits(records: Iterable[dict[str, Any]], key: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for record in records:
        hits = record.get(key) or {}
        for name, count in hits.items():
            out[str(name)] += int(count)
    return out


def has_any(hits: dict[str, Any], names: set[str]) -> bool:
    return any(int(hits.get(name, 0)) > 0 for name in names)


def dominant_failure(record: dict[str, Any]) -> str:
    if record.get("check_pass"):
        return "passes_check"
    if record.get("check_rc") == 124:
        return "compiler_timeout"

    raw_hits = record.get("foreign_syntax_hits") or {}
    candidate_hits = record.get("candidate_foreign_syntax_hits") or {}
    api_hits = record.get("hallucinated_api_hits") or {}
    candidate_api_hits = record.get("candidate_hallucinated_api_hits") or {}
    stderr = str(record.get("check_stderr") or "").lower()

    if has_any(raw_hits, RUST_KEYS) or has_any(candidate_hits, RUST_KEYS):
        return "rust_leakage"
    if "markdown_fence" in raw_hits and not raw_hits.keys() - {"markdown_fence"}:
        return "markdown_fence_or_prose"
    if has_any(raw_hits, FOREIGN_LANG_KEYS) or has_any(candidate_hits, FOREIGN_LANG_KEYS):
        return "foreign_language_leakage"
    if api_hits or candidate_api_hits:
        return "hallucinated_api"
    if "unexpected eof" in stderr or "unterminated" in stderr or "expected" in stderr:
        return "parser_or_truncation"
    if "e200" in stderr or "unknown identifier" in stderr:
        return "unknown_identifier_or_api"
    if "typecheck: failed" in stderr:
        return "sounio_type_or_effect_error"
    if record.get("completion_source") == "error":
        return "generation_or_harness_error"
    return "other_compiler_rejection"


def rows_by_model(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("model_key", "unknown"))].append(record)
    return dict(sorted(grouped.items()))


def category_table(records: list[dict[str, Any]]) -> list[str]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[str(record.get("category", "uncategorized"))].append(record)
    lines = [
        "| Category | Total | Check pass | Candidate syntax pure | Scientific |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, items in sorted(by_category.items()):
        total = len(items)
        passed = sum(1 for item in items if item.get("check_pass"))
        candidate_pure = sum(1 for item in items if item.get("candidate_syntax_pure"))
        sci = "yes" if category in SCIENTIFIC_CATEGORIES else "no"
        lines.append(
            f"| `{category}` | {total} | {passed} ({pct(passed, total)}) | "
            f"{candidate_pure} ({pct(candidate_pure, total)}) | {sci} |"
        )
    return lines


def top_hits_table(counter: Counter[str]) -> list[str]:
    lines = ["| Pattern | Count |", "|---|---:|"]
    if not counter:
        lines.append("| none | 0 |")
        return lines
    for name, count in counter.most_common():
        lines.append(f"| `{name}` | {count} |")
    return lines


def dominant_table(records: list[dict[str, Any]]) -> list[str]:
    counts = Counter(dominant_failure(record) for record in records)
    total = len(records)
    lines = ["| Dominant failure | Count | Rate |", "|---|---:|---:|"]
    for name, count in counts.most_common():
        lines.append(f"| `{name}` | {count} | {pct(count, total)} |")
    return lines


def passing_table(records: list[dict[str, Any]]) -> list[str]:
    passing = [record for record in records if record.get("check_pass")]
    lines = [
        "| Model | Prompt | Category | Run pass | Raw syntax pure | Candidate syntax pure |",
        "|---|---|---|---:|---:|---:|",
    ]
    if not passing:
        lines.append("| none | none | none | n/a | n/a | n/a |")
        return lines
    for record in passing:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{record.get('model_key')}`",
                    f"`{record.get('prompt_id')}`",
                    f"`{record.get('category')}`",
                    str(record.get("run_pass")),
                    str(record.get("syntax_pure")),
                    str(record.get("candidate_syntax_pure")),
                ]
            )
            + " |"
        )
    return lines


def render(records: list[dict[str, Any]], title: str, source: str) -> str:
    by_model = rows_by_model(records)
    run_ids = sorted({str(record.get("run_id", "unknown")) for record in records})
    lines = [
        "<!-- docs:meta",
        "topic_id: repo.docs.training.sounio_ai_eval_failures",
        "authority: repo_only",
        "audience: contributors",
        "last_validated: 2026-05-23",
        "validated_by: Codex",
        "source_of_truth: scripts/dev/sounio_ai_eval_failures.py",
        "-->",
        "",
        f"# {title}",
        "",
        "This report classifies the first public compiler-in-the-loop Sounio AI",
        "eval run into failure modes that can drive the next LoRA repair loop.",
        "",
        "## Source",
        "",
        f"- JSONL: `{source}`",
        f"- Run ids: {', '.join(f'`{run_id}`' for run_id in run_ids)}",
        f"- Records: `{len(records)}`",
        "",
        "## Overall",
        "",
        "| Model | Total | Check pass | Run pass | Raw syntax pure | Candidate syntax pure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, items in by_model.items():
        total = len(items)
        check_pass = sum(1 for item in items if item.get("check_pass"))
        run_attempted = [item for item in items if item.get("run_attempted")]
        run_pass = sum(1 for item in run_attempted if item.get("run_pass"))
        raw_pure = sum(1 for item in items if item.get("syntax_pure"))
        candidate_pure = sum(1 for item in items if item.get("candidate_syntax_pure"))
        run_cell = "n/a" if not run_attempted else f"{run_pass} ({pct(run_pass, len(run_attempted))})"
        lines.append(
            f"| `{model}` | {total} | {check_pass} ({pct(check_pass, total)}) | "
            f"{run_cell} | {raw_pure} ({pct(raw_pure, total)}) | "
            f"{candidate_pure} ({pct(candidate_pure, total)}) |"
        )

    lines.extend(["", "## Passing Candidates", "", *passing_table(records), ""])

    for model, items in by_model.items():
        lines.extend(
            [
                f"## `{model}` Failure Taxonomy",
                "",
                *dominant_table(items),
                "",
                "### Categories",
                "",
                *category_table(items),
                "",
                "### Raw Foreign-Syntax Hits",
                "",
                *top_hits_table(count_hits(items, "foreign_syntax_hits")),
                "",
                "### Candidate Foreign-Syntax Hits",
                "",
                *top_hits_table(count_hits(items, "candidate_foreign_syntax_hits")),
                "",
                "### Hallucinated API Hits",
                "",
                *top_hits_table(count_hits(items, "hallucinated_api_hits")),
                "",
            ]
        )

    lines.extend(
        [
            "## Repair Loop",
            "",
            "The next LoRA should optimize syntax discipline before scientific",
            "competence. The immediate repair dataset should include paired examples",
            "that turn Rust-like, fenced, or prose-heavy outputs into plain Sounio",
            "source files.",
            "",
            "Priority transformations:",
            "",
            "- Remove Markdown fences and explanatory prose from raw output.",
            "- Replace Rust leakage: `println!`, `let mut`, `&mut`, `Vec`, `impl`,",
            "  `.len()`, `.push()`, and semicolon-terminated statements.",
            "- Prefer Sounio idioms: `print`/`println` functions, `var`, `&!T`, fixed",
            "  arrays, explicit effects, and no post-code explanation.",
            "- Keep examples short enough to complete within the generation cap before",
            "  training on long scientific programs.",
            "",
            "Next gate target:",
            "",
            "| Metric | Current | Next target |",
            "|---|---:|---:|",
        "| LoRA compile rate | 2.0% | 10-15% |",
        "| LoRA raw syntax purity | 0.0% | 30%+ |",
        "| LoRA candidate syntax purity | 23.0% | 50%+ |",
        "| Scientific compile rate | 0.0% | nonzero after syntax repair |",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.jsonl)
    markdown = render(records, args.title, args.source_label or str(args.jsonl))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
