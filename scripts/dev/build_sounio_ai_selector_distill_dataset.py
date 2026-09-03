#!/usr/bin/env python3
"""Build selector-distillation records from pass@k compiler-eval results.

The pass@5 gate gives the model several chances and lets `souc check` select
the first compilable candidate. This builder turns those selections into
training records so the next small LoRA can learn to put the compiler-selected
candidate first, without using the fresh v2 eval set as training data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPO_ROOT / "datasets/sounio-ai-eval/prompts.v1.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/sounio-ai-selector-distill/selector_distill.v1.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "datasets/sounio-ai-selector-distill/manifest.v1.json"

SELECTOR_RULES = (
    "Return one valid Sounio source file only. Do not use Markdown. "
    "Prefer the smallest program that satisfies the task and passes `souc check`."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-records", type=int, default=320)
    parser.add_argument("--max-candidate-chars", type=int, default=6000)
    parser.add_argument(
        "--prefer-run-pass",
        action="store_true",
        help="Prefer the first runtime-passing candidate over the first check-passing candidate.",
    )
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


def load_prompts(path: Path) -> dict[str, dict[str, Any]]:
    prompts: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("id"):
            prompts[str(row["id"])] = row
    return prompts


def read_source(path_text: str, max_chars: int) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text or len(text) > max_chars:
        return ""
    return text + "\n"


def choose_candidate(rows: list[dict[str, Any]], prefer_run_pass: bool) -> dict[str, Any] | None:
    rows = sorted(rows, key=lambda row: int(row.get("sample_index", 0)))
    if prefer_run_pass:
        for row in rows:
            if row.get("run_pass"):
                return row
    for row in rows:
        if row.get("check_pass"):
            return row
    return None


def candidate_label(row: dict[str, Any]) -> str:
    sample = int(row.get("sample_index", 0))
    check = "pass" if row.get("check_pass") else "fail"
    run = row.get("run_pass")
    if run is True:
        run_text = "run_pass"
    elif run is False:
        run_text = "run_fail"
    else:
        run_text = "run_not_attempted"
    syntax = "syntax_pure" if row.get("candidate_syntax_pure") else "syntax_impure"
    return f"sample{sample}: check_{check}, {run_text}, {syntax}"


def compact_candidate(row: dict[str, Any], max_chars: int) -> str:
    source = read_source(str(row.get("completion_path") or ""), max_chars)
    if not source:
        source = read_source(str(row.get("raw_completion_path") or ""), max_chars)
    return source.strip()


def record_id(prefix: str, prompt_id: str, sample: int, index: int) -> str:
    safe_prompt = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in prompt_id)
    return f"selector_distill_v1_{index:04d}_{prefix}_sample{sample}_{safe_prompt}"[:180]


def make_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = load_prompts(args.prompts)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ignored_models: Counter[str] = Counter()
    for row in read_jsonl(args.results_jsonl):
        model_key = str(row.get("model_key", ""))
        if model_key != args.model_key:
            ignored_models[model_key] += 1
            continue
        grouped[str(row.get("prompt_id", ""))].append(row)

    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    selected_samples: Counter[int] = Counter()
    categories: Counter[str] = Counter()

    def add_record(
        *,
        kind: str,
        prompt: dict[str, Any],
        selected: dict[str, Any],
        output: str,
        input_text: str,
    ) -> None:
        if len(records) >= args.max_records:
            return
        prompt_id = str(prompt.get("id") or selected.get("prompt_id"))
        sample_index = int(selected.get("sample_index", 0))
        records.append(
            {
                "id": record_id(kind, prompt_id, sample_index, len(records) + 1),
                "instruction": (
                    f"{SELECTOR_RULES}\n\n"
                    "Selector-distillation task: produce as sample0 the candidate "
                    "that the Sounio compiler selector would choose."
                ),
                "input": input_text,
                "output": output,
                "good": output,
                "bad": input_text,
                "rule": kind,
                "repair_kind": kind,
                "category": str(prompt.get("category") or selected.get("category") or "uncategorized"),
                "difficulty": str(prompt.get("difficulty") or selected.get("difficulty") or "selector_distill"),
                "tags": sorted(
                    set(
                        list(prompt.get("tags") or [])
                        + list(selected.get("tags") or [])
                        + [
                            "selector_distill",
                            "compiler_selected",
                            f"selected_sample_{sample_index}",
                            "run_pass" if selected.get("run_pass") else "check_pass",
                        ]
                    )
                ),
                "source_path": str(prompt.get("source_path") or selected.get("source_path") or ""),
                "source_prompt_id": prompt_id,
                "from_eval_failure": False,
                "selector_model_key": args.model_key,
                "selected_sample_index": sample_index,
                "selected_check_pass": bool(selected.get("check_pass")),
                "selected_run_pass": bool(selected.get("run_pass")),
                "source_eval_run": selected.get("run_id"),
            }
        )

    for prompt_id, rows in sorted(grouped.items()):
        if not prompt_id:
            skipped["missing_prompt_id"] += 1
            continue
        prompt = prompts.get(prompt_id)
        if prompt is None:
            skipped["prompt_not_found"] += 1
            continue
        selected = choose_candidate(rows, args.prefer_run_pass)
        if selected is None:
            skipped["no_passing_candidate"] += 1
            continue
        output = compact_candidate(selected, args.max_candidate_chars)
        if not output:
            skipped["selected_source_missing"] += 1
            continue
        selected_samples[int(selected.get("sample_index", 0))] += 1
        categories[str(prompt.get("category") or selected.get("category") or "uncategorized")] += 1

        task_text = str(prompt.get("prompt") or "").strip()
        source_path = str(prompt.get("source_path") or selected.get("source_path") or "")
        base_input = (
            f"Original task id: {prompt_id}\n"
            f"Category: {prompt.get('category', selected.get('category', 'uncategorized'))}\n"
            f"Source path: {source_path}\n\n"
            f"Task:\n{task_text}\n"
        ).strip()
        add_record(
            kind="selector_selected_to_sample0",
            prompt=prompt,
            selected=selected,
            output=output,
            input_text=base_input,
        )

        prior_rows = [
            row
            for row in sorted(rows, key=lambda item: int(item.get("sample_index", 0)))
            if int(row.get("sample_index", 0)) < int(selected.get("sample_index", 0))
        ]
        if prior_rows:
            snippets: list[str] = []
            for row in prior_rows[:3]:
                candidate = compact_candidate(row, min(args.max_candidate_chars, 1600))
                if not candidate:
                    continue
                snippets.append(f"{candidate_label(row)}\n```sounio\n{candidate}\n```")
            if snippets:
                add_record(
                    kind="failed_candidates_to_selected",
                    prompt=prompt,
                    selected=selected,
                    output=output,
                    input_text=(
                        f"{base_input}\n\n"
                        "Earlier candidates did not win the compiler selector:\n\n"
                        + "\n\n".join(snippets)
                    ),
                )

        if selected.get("run_pass"):
            add_record(
                kind="runtime_selected_to_sample0",
                prompt=prompt,
                selected=selected,
                output=output,
                input_text=(
                    f"{base_input}\n\n"
                    "This output should compile and run under the eval harness."
                ),
            )

    manifest = {
        "name": "sounio-ai-selector-distill",
        "version": "v1",
        "description": "Compiler-selector distillation records built from pass@k eval candidates.",
        "results_jsonl": str(args.results_jsonl),
        "prompts": str(args.prompts),
        "model_key": args.model_key,
        "prefer_run_pass": args.prefer_run_pass,
        "record_count": len(records),
        "prompt_count_with_model_rows": len(grouped),
        "selected_prompt_count": sum(selected_samples.values()),
        "selected_sample_counts": selected_samples,
        "category_counts": categories,
        "rule_counts": Counter(record["rule"] for record in records),
        "skipped": skipped,
        "ignored_model_counts": ignored_models,
    }
    return records, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_records <= 0:
        raise SystemExit("--max-records must be positive")
    records, manifest = make_records(args)
    if not records:
        raise SystemExit("no selector-distillation records generated")
    write_jsonl(args.output, records)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} records to {args.output}")
    print(f"wrote manifest to {args.manifest}")
    print("rules:", dict(Counter(record["rule"] for record in records)))


if __name__ == "__main__":
    main()
