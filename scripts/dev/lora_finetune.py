#!/usr/bin/env python3
"""LoRA fine-tune for Sounio syntax knowledge.

Default model: Qwen/Qwen2.5-Coder-1.5B
Method: PEFT LoRA on q_proj + v_proj, rank=16
Dataset: contrastive, code examples, and the syntax-repair corpus

Usage:
    pip install transformers peft datasets torch accelerate
    python3 scripts/dev/lora_finetune.py [--model Qwen/Qwen2.5-Coder-1.5B]
                                          [--output ./lora-sounio]
                                          [--epochs 3]
                                          [--batch-size 4]
                                          [--target-records 480]
                                          [--dry-run]

Dry-run (no GPU needed):
    python3 scripts/dev/lora_finetune.py --dry-run
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = [
    REPO_ROOT / "datasets/sounio-ai-mandelbrot-d2-canonical/mandelbrot_d2_canonical.v1.jsonl",
    REPO_ROOT / "datasets/sounio-contrastive/contrastive.jsonl",
    REPO_ROOT / "datasets/sounio-contrastive/synthetic.jsonl",
    REPO_ROOT / "datasets/sounio-code-examples/train.jsonl",
    REPO_ROOT / "datasets/sounio-ai-structure-hybrid/structure_hybrid.v1.jsonl",
    REPO_ROOT / "datasets/sounio-ai-repair/repair.v1.jsonl",
    REPO_ROOT / "datasets/sounio-ai-repair/repair.v2-heldout.jsonl",
    REPO_ROOT / "datasets/sounio-ai-science-repair/science_repair.v1.jsonl",
    REPO_ROOT / "datasets/sounio-ai-selector-distill/selector_distill.v1.jsonl",
    REPO_ROOT / "datasets/sounio-ai-octonion-repair/octonion_repair.v1.jsonl",
]

SYSTEM_PROMPT = """You are a Sounio programming assistant. Sounio is a systems language that is NOT Rust.

Critical rules:
- No semicolons (statements end at newline)
- `var` for mutable bindings (not `let mut`)
- `&!T` for exclusive/mutable references (not `&mut T`)
- No macros: use `assert()` not `assert!()`, `println()` not `println!()`
- Negative numbers: write `0 - 42` not `-42`
- Functions must declare all effects: `fn f() with IO { ... }`
- No generics except `Knowledge<T>` — use fixed-size arrays `[T; N]`
- No closures — use named function references

Write only valid Sounio code."""

SOURCE_ONLY_PROMPT = """Return raw Sounio source code only.
Do not write an introduction, explanation, Markdown fence, language label, or postscript.
The first generated line must be Sounio code or a Sounio comment."""


MAX_CHARS = 6000  # ~1500 tokens — filter out files that are too long


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def load_dataset() -> list[dict]:
    """Load and merge all datasets into a unified format."""
    records = []

    for path in DATASETS:
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        for r in read_jsonl(path):
            dataset_name = str(path.relative_to(REPO_ROOT))

            # Normalize to {instruction, input, output} format.
            if "completion" in r:
                inp = r.get("input", "")
                out = r.get("completion", "")
                if len(inp) + len(out) > MAX_CHARS:
                    continue
                records.append({
                    "instruction": r.get("instruction", "Write correct Sounio code:"),
                    "input": inp,
                    "output": out,
                    "rule": r.get("rule", ""),
                    "source_path": r.get("source_path", ""),
                    "source_prompt_id": r.get("source_prompt_id", ""),
                    "dataset": dataset_name,
                })
            elif "instruction" in r and "output" in r:
                inp = r.get("input", "")
                out = r.get("output", "")
                if len(inp) + len(out) <= MAX_CHARS:
                    normalized = dict(r)
                    normalized.setdefault("source_prompt_id", r.get("source_prompt_id", ""))
                    normalized.setdefault("source_path", r.get("source_path", ""))
                    normalized["dataset"] = dataset_name
                    records.append(normalized)

    print(f"Loaded {len(records)} total records")
    return records


def load_heldout_prompts(paths: list[Path]) -> tuple[set[str], set[str]]:
    prompt_ids: set[str] = set()
    source_paths: set[str] = set()
    for path in paths:
        for record in read_jsonl(path):
            if record.get("id"):
                prompt_ids.add(str(record["id"]))
            if record.get("source_path"):
                source_paths.add(str(record["source_path"]))
    return prompt_ids, source_paths


def filter_heldout(
    records: list[dict],
    heldout_prompt_ids: set[str],
    heldout_source_paths: set[str],
) -> tuple[list[dict], list[dict]]:
    if not heldout_prompt_ids and not heldout_source_paths:
        return records, []
    kept: list[dict] = []
    excluded: list[dict] = []
    for record in records:
        source_prompt_id = str(record.get("source_prompt_id") or record.get("prompt_id") or "")
        source_path = str(record.get("source_path") or "")
        if source_prompt_id in heldout_prompt_ids or source_path in heldout_source_paths:
            excluded.append(record)
        else:
            kept.append(record)
    return kept, excluded


def write_train_manifest(
    path: Path,
    *,
    records: list[dict],
    sampled: list[dict],
    excluded: list[dict],
    heldout_prompt_ids: set[str],
    heldout_source_paths: set[str],
    target_records: int,
    seed: int,
    prompt_format: str,
    science_record_fraction: float,
    dataset_mode: str,
) -> None:
    manifest = {
        "eligible_after_heldout_filter": len(records),
        "excluded_by_heldout_count": len(excluded),
        "heldout_prompt_count": len(heldout_prompt_ids),
        "heldout_source_path_count": len(heldout_source_paths),
        "sampled_count": len(sampled),
        "target_records": target_records,
        "seed": seed,
        "prompt_format": prompt_format,
        "science_record_fraction": science_record_fraction,
        "dataset_mode": dataset_mode,
        "sampled_dataset_counts": Counter(r.get("dataset", "<unknown>") for r in sampled),
        "sampled_rule_counts": Counter(r.get("rule", "full_program") for r in sampled),
        "sampled_category_counts": Counter(r.get("category", "<unknown>") for r in sampled),
        "excluded_dataset_counts": Counter(r.get("dataset", "<unknown>") for r in excluded),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_science_structure_record(record: dict) -> bool:
    if "sounio-ai-science-repair" in str(record.get("dataset", "")):
        return True
    tags = record.get("tags") or []
    return "science_structure" in tags


def is_selector_distill_record(record: dict) -> bool:
    if "sounio-ai-selector-distill" in str(record.get("dataset", "")):
        return True
    tags = record.get("tags") or []
    return "selector_distill" in tags


def is_structure_hybrid_record(record: dict) -> bool:
    if "sounio-ai-structure-hybrid" in str(record.get("dataset", "")):
        return True
    tags = record.get("tags") or []
    return "structure_hybrid" in tags


def is_octonion_repair_record(record: dict) -> bool:
    if "sounio-ai-octonion-repair" in str(record.get("dataset", "")):
        return True
    tags = record.get("tags") or []
    return "octonion_repair" in tags



def is_mandelbrot_d2_canonical_record(record: dict) -> bool:
    if "sounio-ai-mandelbrot-d2-canonical" in str(record.get("dataset", "")):
        return True
    tags = record.get("tags") or []
    return "canonical_structural_oracle" in tags

def stratified_sample(
    records: list[dict],
    target: int = 2000,
    science_record_fraction: float = 0.0,
    dataset_mode: str = "default",
) -> list[dict]:
    """
    Stratify without duplicating records across buckets.

    Target shape:
    - 35% contrastive/repair rules
    - 30% source-backed examples
    - 20% full programs
    - 15% effects/types
    """
    if target <= 0:
        raise ValueError("target must be positive")
    if len(records) <= target and dataset_mode == "default":
        sampled = records[:]
        random.shuffle(sampled)
        return sampled

    remaining = records[:]
    sampled: list[dict] = []

    if dataset_mode == "selector_distill":
        selector_candidates = [r for r in remaining if is_selector_distill_record(r)]
        science_candidates = [r for r in remaining if is_science_structure_record(r) and not is_selector_distill_record(r)]
        guard_candidates = [
            r for r in remaining if not is_selector_distill_record(r) and not is_science_structure_record(r)
        ]
        selector_quota = int(target * 0.55)
        science_quota = int(target * 0.25)
        if selector_candidates:
            chosen = (
                selector_candidates
                if len(selector_candidates) <= selector_quota
                else random.sample(selector_candidates, selector_quota)
            )
            sampled.extend(chosen)
        need_science = min(science_quota, target - len(sampled))
        if need_science > 0 and science_candidates:
            chosen = (
                science_candidates
                if len(science_candidates) <= need_science
                else random.sample(science_candidates, need_science)
            )
            sampled.extend(chosen)
        need = target - len(sampled)
        if need > 0 and guard_candidates:
            sampled.extend(_bucket_sample(guard_candidates, need))
        if len(sampled) < target:
            chosen_ids = {id(r) for r in sampled}
            leftovers = [r for r in records if id(r) not in chosen_ids]
            if leftovers:
                sampled.extend(random.sample(leftovers, min(target - len(sampled), len(leftovers))))
        random.shuffle(sampled)
        return sampled[:target]

    if dataset_mode == "selector_only":
        selector_candidates = [r for r in remaining if is_selector_distill_record(r)]
        if not selector_candidates:
            return []
        if len(selector_candidates) <= target:
            sampled = selector_candidates[:]
        else:
            sampled = random.sample(selector_candidates, target)
        random.shuffle(sampled)
        return sampled

    if dataset_mode == "structure_hybrid":
        hybrid_candidates = [r for r in remaining if is_structure_hybrid_record(r)]
        guard_candidates = [r for r in remaining if not is_structure_hybrid_record(r)]
        hybrid_quota = int(target * 0.80)
        if hybrid_candidates:
            chosen = (
                hybrid_candidates
                if len(hybrid_candidates) <= hybrid_quota
                else random.sample(hybrid_candidates, hybrid_quota)
            )
            sampled.extend(chosen)
        need = target - len(sampled)
        if need > 0 and guard_candidates:
            sampled.extend(_bucket_sample(guard_candidates, need))
        if len(sampled) < target:
            chosen_ids = {id(r) for r in sampled}
            leftovers = [r for r in records if id(r) not in chosen_ids]
            if leftovers:
                sampled.extend(random.sample(leftovers, min(target - len(sampled), len(leftovers))))
        random.shuffle(sampled)
        return sampled[:target]

    if dataset_mode == "structure_selector_focus":
        hybrid_candidates = [r for r in remaining if is_structure_hybrid_record(r)]
        selector_candidates = [r for r in remaining if is_selector_distill_record(r)]
        science_candidates = [
            r for r in remaining
            if is_science_structure_record(r)
            and not is_structure_hybrid_record(r)
            and not is_selector_distill_record(r)
        ]
        guard_candidates = [
            r for r in remaining
            if not is_structure_hybrid_record(r)
            and not is_selector_distill_record(r)
            and not is_science_structure_record(r)
        ]
        for candidates, quota in [
            (selector_candidates, max(1, int(target * 0.50))),
            (hybrid_candidates, max(1, int(target * 0.25))),
            (science_candidates, max(0, int(target * 0.15))),
        ]:
            if not candidates or len(sampled) >= target:
                continue
            need = min(quota, target - len(sampled))
            chosen = candidates if len(candidates) <= need else random.sample(candidates, need)
            sampled.extend(chosen)
        need = target - len(sampled)
        if need > 0 and guard_candidates:
            sampled.extend(_bucket_sample(guard_candidates, need))
        if len(sampled) < target:
            chosen_ids = {id(r) for r in sampled}
            leftovers = [r for r in records if id(r) not in chosen_ids]
            if leftovers:
                sampled.extend(random.sample(leftovers, min(target - len(sampled), len(leftovers))))
        random.shuffle(sampled)
        return sampled[:target]

    if dataset_mode == "structure_conservative":
        hybrid_candidates = [r for r in remaining if is_structure_hybrid_record(r)]
        guard_candidates = [r for r in remaining if not is_structure_hybrid_record(r)]
        hybrid_quota = max(1, int(target * 0.05))
        if hybrid_candidates:
            chosen = (
                hybrid_candidates
                if len(hybrid_candidates) <= hybrid_quota
                else random.sample(hybrid_candidates, hybrid_quota)
            )
            sampled.extend(chosen)
        need = target - len(sampled)
        if need > 0 and guard_candidates:
            sampled.extend(_bucket_sample(guard_candidates, need))
        if len(sampled) < target:
            chosen_ids = {id(r) for r in sampled}
            leftovers = [r for r in records if id(r) not in chosen_ids]
            if leftovers:
                sampled.extend(random.sample(leftovers, min(target - len(sampled), len(leftovers))))
        random.shuffle(sampled)
        return sampled[:target]

    if dataset_mode in {"substance_anchor_heavy", "substance_anchor_balanced"}:
        oct_candidates = [r for r in remaining if is_octonion_repair_record(r)]
        structure_candidates = [r for r in remaining if is_structure_hybrid_record(r)]
        science_candidates = [r for r in remaining if is_science_structure_record(r)]
        long_source_candidates = [
            r for r in remaining
            if r.get("source_path")
            and len(r.get("output", "")) >= 260
            and not is_octonion_repair_record(r)
            and not is_structure_hybrid_record(r)
            and not is_science_structure_record(r)
        ]
        if dataset_mode == "substance_anchor_heavy":
            quotas = [
                (oct_candidates, max(1, int(target * 0.70))),
                (science_candidates, max(1, int(target * 0.15))),
                (structure_candidates, max(1, int(target * 0.10))),
                (long_source_candidates, max(1, int(target * 0.05))),
            ]
        else:
            quotas = [
                (oct_candidates, max(1, int(target * 0.60))),
                (science_candidates, max(1, int(target * 0.20))),
                (structure_candidates, max(1, int(target * 0.10))),
                (long_source_candidates, max(1, int(target * 0.10))),
            ]
        for candidates, quota in quotas:
            if not candidates or len(sampled) >= target:
                continue
            need = min(quota, target - len(sampled))
            chosen = candidates if len(candidates) <= need else random.sample(candidates, need)
            sampled.extend(chosen)
        if len(sampled) < target:
            chosen_ids = {id(r) for r in sampled}
            leftovers = [
                r for r in records
                if id(r) not in chosen_ids
                and (is_octonion_repair_record(r) or len(r.get("output", "")) >= 180)
            ]
            if leftovers:
                sampled.extend(random.sample(leftovers, min(target - len(sampled), len(leftovers))))
        random.shuffle(sampled)
        return sampled[:target]

    if dataset_mode == "mandelbrot_d2_canonical":
        md2_candidates = [r for r in remaining if is_mandelbrot_d2_canonical_record(r)]
        oct_candidates = [r for r in remaining if is_octonion_repair_record(r)]
        science_candidates = [r for r in remaining if is_science_structure_record(r)]
        structure_candidates = [r for r in remaining if is_structure_hybrid_record(r)]
        for candidates, quota in [
            (md2_candidates, max(1, int(target * 0.45))),
            (oct_candidates, max(1, int(target * 0.30))),
            (science_candidates, max(1, int(target * 0.20))),
            (structure_candidates, max(1, int(target * 0.05))),
        ]:
            if not candidates or len(sampled) >= target:
                continue
            need = min(quota, target - len(sampled))
            chosen = candidates if len(candidates) <= need else random.sample(candidates, need)
            sampled.extend(chosen)
        if len(sampled) < target:
            chosen_ids = {id(r) for r in sampled}
            leftovers = [r for r in records if id(r) not in chosen_ids and len(r.get("output", "")) >= 180]
            if leftovers:
                sampled.extend(random.sample(leftovers, min(target - len(sampled), len(leftovers))))
        random.shuffle(sampled)
        return sampled[:target]

    if dataset_mode == "substance_recover":
        oct_candidates = [r for r in remaining if is_octonion_repair_record(r)]
        structure_candidates = [r for r in remaining if is_structure_hybrid_record(r)]
        science_candidates = [r for r in remaining if is_science_structure_record(r)]
        long_source_candidates = [
            r for r in remaining
            if r.get("source_path")
            and len(r.get("output", "")) >= 260
            and not is_octonion_repair_record(r)
            and not is_structure_hybrid_record(r)
            and not is_science_structure_record(r)
        ]
        generic_long_candidates = [
            r for r in remaining
            if len(r.get("output", "")) >= 260
            and not is_octonion_repair_record(r)
            and not is_structure_hybrid_record(r)
            and not is_science_structure_record(r)
            and r not in long_source_candidates
        ]
        for candidates, quota in [
            (oct_candidates, max(1, int(target * 0.25))),
            (structure_candidates, max(1, int(target * 0.25))),
            (science_candidates, max(1, int(target * 0.25))),
            (long_source_candidates, max(1, int(target * 0.20))),
            (generic_long_candidates, max(0, int(target * 0.05))),
        ]:
            if not candidates or len(sampled) >= target:
                continue
            need = min(quota, target - len(sampled))
            chosen = candidates if len(candidates) <= need else random.sample(candidates, need)
            sampled.extend(chosen)
        if len(sampled) < target:
            chosen_ids = {id(r) for r in sampled}
            leftovers = [r for r in records if id(r) not in chosen_ids and len(r.get("output", "")) >= 180]
            if leftovers:
                sampled.extend(random.sample(leftovers, min(target - len(sampled), len(leftovers))))
        random.shuffle(sampled)
        return sampled[:target]

    if dataset_mode == "octonion_repair":
        oct_candidates = [r for r in remaining if is_octonion_repair_record(r)]
        structure_candidates = [
            r for r in remaining
            if is_structure_hybrid_record(r) and not is_octonion_repair_record(r)
        ]
        selector_candidates = [
            r for r in remaining
            if is_selector_distill_record(r) and not is_octonion_repair_record(r)
        ]
        guard_candidates = [
            r for r in remaining
            if not is_octonion_repair_record(r)
            and not is_structure_hybrid_record(r)
            and not is_selector_distill_record(r)
        ]
        for candidates, quota in [
            (oct_candidates, max(1, int(target * 0.55))),
            (structure_candidates, max(1, int(target * 0.20))),
            (selector_candidates, max(0, int(target * 0.15))),
        ]:
            if not candidates or len(sampled) >= target:
                continue
            need = min(quota, target - len(sampled))
            chosen = candidates if len(candidates) <= need else random.sample(candidates, need)
            sampled.extend(chosen)
        need = target - len(sampled)
        if need > 0 and guard_candidates:
            sampled.extend(_bucket_sample(guard_candidates, need))
        if len(sampled) < target:
            chosen_ids = {id(r) for r in sampled}
            leftovers = [r for r in records if id(r) not in chosen_ids]
            if leftovers:
                sampled.extend(random.sample(leftovers, min(target - len(sampled), len(leftovers))))
        random.shuffle(sampled)
        return sampled[:target]

    if dataset_mode in {"science_only", "science_balanced"}:
        science_candidates = [r for r in remaining if is_science_structure_record(r)]
        guard_candidates = [r for r in remaining if not is_science_structure_record(r)]
        if dataset_mode == "science_only":
            if len(science_candidates) <= target:
                sampled = science_candidates[:]
            else:
                sampled = random.sample(science_candidates, target)
            random.shuffle(sampled)
            return sampled

        science_quota = max(int(target * science_record_fraction), int(target * 0.70))
        science_quota = min(science_quota, target)
        if science_candidates:
            chosen = (
                science_candidates
                if len(science_candidates) <= science_quota
                else random.sample(science_candidates, science_quota)
            )
            sampled.extend(chosen)
        need = target - len(sampled)
        if need > 0 and guard_candidates:
            sampled.extend(_bucket_sample(guard_candidates, need))
        if len(sampled) < target:
            chosen_ids = {id(r) for r in sampled}
            leftovers = [r for r in records if id(r) not in chosen_ids]
            if leftovers:
                sampled.extend(random.sample(leftovers, min(target - len(sampled), len(leftovers))))
        random.shuffle(sampled)
        return sampled[:target]

    if science_record_fraction > 0.0:
        science_candidates = [r for r in remaining if is_science_structure_record(r)]
        science_quota = int(target * science_record_fraction)
        if science_candidates and science_quota > 0:
            chosen = (
                science_candidates
                if len(science_candidates) <= science_quota
                else random.sample(science_candidates, science_quota)
            )
            chosen_ids = {id(r) for r in chosen}
            sampled.extend(chosen)
            remaining = [r for r in remaining if id(r) not in chosen_ids]

    sampled.extend(_bucket_sample(remaining, target - len(sampled)))
    random.shuffle(sampled)
    return sampled[:target]


def _bucket_sample(records: list[dict], target: int) -> list[dict]:
    if target <= 0:
        return []
    if len(records) <= target:
        sampled = records[:]
        random.shuffle(sampled)
        return sampled

    remaining = records[:]
    sampled: list[dict] = []
    bucket_specs = [
        ("contrastive_or_repair", 0.35, lambda r: bool(r.get("rule"))),
        ("source_backed", 0.30, lambda r: bool(r.get("source_path"))),
        ("full_program", 0.20, lambda r: len(r.get("output", "")) > 300),
        ("effects", 0.15, lambda r: "with " in r.get("output", "")),
    ]
    remaining_target = target
    for _name, fraction, pred in bucket_specs:
        if remaining_target <= 0:
            break
        candidates = [r for r in remaining if pred(r)]
        quota = int(remaining_target * fraction)
        if not candidates or quota <= 0:
            continue
        chosen = candidates if len(candidates) <= quota else random.sample(candidates, quota)
        chosen_ids = {id(r) for r in chosen}
        sampled.extend(chosen)
        remaining = [r for r in remaining if id(r) not in chosen_ids]

    need = target - len(sampled)
    if need > 0 and remaining:
        sampled.extend(random.sample(remaining, min(need, len(remaining))))

    random.shuffle(sampled)
    return sampled


def format_alpaca(r: dict) -> str:
    """Format a record as Alpaca-style instruction prompt."""
    if r.get("input", "").strip():
        return (
            f"### System\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction\n{r['instruction']}\n\n"
            f"### Input\n{r['input']}\n\n"
            f"### Response\n{r['output']}"
        )
    return (
        f"### System\n{SYSTEM_PROMPT}\n\n"
        f"### Instruction\n{r['instruction']}\n\n"
        f"### Response\n{r['output']}"
    )


def format_source_only(r: dict) -> str:
    """Format a record to match `--prompt-style source_only` at eval time."""
    task_parts = [str(r["instruction"]).strip()]
    if r.get("input", "").strip():
        task_parts.extend(["", str(r["input"]).strip()])
    task = "\n".join(task_parts).strip()
    return (
        f"{SOURCE_ONLY_PROMPT}\n\n"
        f"Task:\n{task}\n\n"
        f"Begin raw Sounio source now:\n{r['output']}"
    )


def format_record(r: dict, prompt_format: str) -> str:
    if prompt_format == "alpaca":
        return format_alpaca(r)
    if prompt_format == "source_only":
        return format_source_only(r)
    raise ValueError(f"unknown prompt format: {prompt_format}")


def run_dry(
    records: list[dict],
    target_records: int,
    train_manifest: Path | None,
    excluded: list[dict],
    heldout_prompt_ids: set[str],
    heldout_source_paths: set[str],
    seed: int,
    prompt_format: str,
    science_record_fraction: float,
    dataset_mode: str,
):
    """Validate dataset without GPU."""
    print("\n--- Dry run: dataset validation ---")
    sampled = stratified_sample(records, target_records, science_record_fraction, dataset_mode)
    print(f"Stratified sample: {len(sampled)} records")
    print(f"Held-out excluded records: {len(excluded)}")

    # Show distribution
    rules = [r.get("rule", "full_program") for r in sampled]
    for rule, count in Counter(rules).most_common(10):
        print(f"  {rule or 'full_program'}: {count}")

    # Show one formatted example
    print("\n--- Example training record ---")
    ex = sampled[0]
    formatted = format_record(ex, prompt_format)
    print(formatted[:800])
    print("...\n")

    # Token length estimate (rough: 4 chars per token)
    lengths = [len(format_record(r, prompt_format)) // 4 for r in sampled]
    avg = sum(lengths) / len(lengths)
    max_len = max(lengths)
    print(f"Token length estimate: avg={avg:.0f}, max={max_len}")
    if max_len > 2048:
        print(f"  WARNING: {sum(1 for l in lengths if l > 2048)} records exceed 2048 tokens")
    if train_manifest is not None:
        write_train_manifest(
            train_manifest,
            records=records,
            sampled=sampled,
            excluded=excluded,
            heldout_prompt_ids=heldout_prompt_ids,
            heldout_source_paths=heldout_source_paths,
            target_records=target_records,
            seed=seed,
            prompt_format=prompt_format,
            science_record_fraction=science_record_fraction,
            dataset_mode=dataset_mode,
        )
        print(f"Wrote training manifest: {train_manifest}")

    print("\nDry run complete. Run without --dry-run on a GPU machine to train.")


def run_train(
    records: list[dict],
    model_name: str,
    adapter_init: str | None,
    output_dir: str,
    epochs: int,
    batch_size: int,
    target_records: int,
    max_length: int,
    learning_rate: float,
    load_in_4bit: bool,
    train_manifest: Path | None,
    excluded: list[dict],
    heldout_prompt_ids: set[str],
    heldout_source_paths: set[str],
    seed: int,
    prompt_format: str,
    science_record_fraction: float,
    dataset_mode: str,
):
    """Run LoRA fine-tuning."""
    try:
        import inspect
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig, PeftModel, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install transformers peft datasets torch accelerate")
        sys.exit(1)

    sampled = stratified_sample(records, target_records, science_record_fraction, dataset_mode)
    print(f"Training on {len(sampled)} records")
    print(f"Held-out excluded records: {len(excluded)}")
    if train_manifest is not None:
        write_train_manifest(
            train_manifest,
            records=records,
            sampled=sampled,
            excluded=excluded,
            heldout_prompt_ids=heldout_prompt_ids,
            heldout_source_paths=heldout_source_paths,
            target_records=target_records,
            seed=seed,
            prompt_format=prompt_format,
            science_record_fraction=science_record_fraction,
            dataset_mode=dataset_mode,
        )
        print(f"Wrote training manifest: {train_manifest}")

    # Tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format dataset
    texts = [format_record(r, prompt_format) for r in sampled]
    split = int(len(texts) * 0.9)
    train_texts, val_texts = texts[:split], texts[split:]

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_ds = Dataset.from_dict({"text": train_texts}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts}).map(tokenize, batched=True)
    train_ds = train_ds.map(lambda x: {"labels": x["input_ids"]})
    val_ds = val_ds.map(lambda x: {"labels": x["input_ids"]})

    # Model + LoRA
    print(f"Loading model: {model_name}")
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if adapter_init:
        print(f"Continuing from adapter: {adapter_init}")
        model = PeftModel.from_pretrained(model, adapter_init, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training
    training_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "warmup_ratio": 0.05,
        "learning_rate": learning_rate,
        "fp16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "report_to": "none",
    }
    training_args_sig = inspect.signature(TrainingArguments)
    if "eval_strategy" in training_args_sig.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Training...")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved to {output_dir}")
    print("Load with: model = PeftModel.from_pretrained(base_model, '{output_dir}')")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune for Sounio syntax")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B",
                        help="Base model (HuggingFace model ID)")
    parser.add_argument(
        "--adapter-init",
        help="Optional existing PEFT adapter directory to continue training from.",
    )
    parser.add_argument("--output", default="./lora-sounio",
                        help="Output directory for LoRA weights")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-records", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--prompt-format",
        choices=["alpaca", "source_only"],
        default="alpaca",
        help="Training prompt format. source_only matches the eval-time source_only prompt style.",
    )
    parser.add_argument(
        "--science-record-fraction",
        type=float,
        default=0.0,
        help="Reserve this fraction of the sampled training set for science-structure records.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=[
            "default",
            "science_balanced",
            "science_only",
            "selector_distill",
            "selector_only",
            "structure_hybrid",
            "structure_selector_focus",
            "structure_conservative",
            "octonion_repair",
            "substance_recover",
            "mandelbrot_d2_canonical",
            "substance_anchor_heavy",
            "substance_anchor_balanced",
        ],
        default="default",
        help=(
            "Sampling mode. science_balanced emphasizes compiler-checked science "
            "records and fills the rest with syntax guards; science_only excludes "
            "non-science records; selector_distill prioritizes compiler-selector "
            "distillation records with science/guard fill; selector_only uses only "
            "compiler-selector distillation records; structure_hybrid prioritizes v13 octonion/O-SSM "
            "compiler-backed structure records; structure_selector_focus mixes "
            "selector distillation with octonion/O-SSM structure records; "
            "structure_conservative keeps the same structure records as a small regularizer; "
            "octonion_repair prioritizes compiler-checked nonheldout octonion/Fano repair records."
        ),
    )
    parser.add_argument(
        "--heldout-prompts",
        action="append",
        type=Path,
        default=[],
        help="Prompt JSONL to exclude from training by prompt id and source_path. Repeatable.",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        help="Write a JSON manifest describing held-out exclusions and sampled records.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit loading. Off by default for the small Qwen repair gate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate dataset without GPU or model download")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading datasets...")
    records = load_dataset()
    heldout_prompt_ids, heldout_source_paths = load_heldout_prompts(args.heldout_prompts)
    if args.heldout_prompts:
        print(
            f"Loaded held-out prompts: {len(heldout_prompt_ids)} ids, "
            f"{len(heldout_source_paths)} source paths"
        )
    records, excluded = filter_heldout(records, heldout_prompt_ids, heldout_source_paths)
    if excluded:
        print(f"Excluded {len(excluded)} training records that overlap held-out prompts")
    if not records:
        print("No records loaded. Check dataset paths.")
        sys.exit(1)

    if args.dry_run:
        run_dry(
            records,
            args.target_records,
            args.train_manifest,
            excluded,
            heldout_prompt_ids,
            heldout_source_paths,
            args.seed,
            args.prompt_format,
            args.science_record_fraction,
            args.dataset_mode,
        )
    else:
        run_train(
            records,
            args.model,
            args.adapter_init,
            args.output,
            args.epochs,
            args.batch_size,
            args.target_records,
            args.max_length,
            args.learning_rate,
            args.load_in_4bit,
            args.train_manifest,
            excluded,
            heldout_prompt_ids,
            heldout_source_paths,
            args.seed,
            args.prompt_format,
            args.science_record_fraction,
            args.dataset_mode,
        )


if __name__ == "__main__":
    main()
