#!/usr/bin/env python3
"""Build v13i octonion/Fano repair records from non-heldout sources."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "datasets/sounio-ai-octonion-repair/octonion_repair.v1.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "datasets/sounio-ai-octonion-repair/manifest.v1.json"
DEFAULT_HELDOUT = REPO_ROOT / "datasets/sounio-ai-eval/prompts.v2.jsonl"

SOURCE_CANDIDATES = [
    "tests/native-v2/science_spine/octonion_fano_168.sio",
    "tests/native-v2/science_spine/q_deform_octonion.sio",
    "tests/native-v2/science_spine/knowledge_octonion_variance.sio",
    "tests/run-pass/fano_basics.sio",
    "tests/run-pass/embed_octonion.sio",
    "tests/run-pass/octonion_basic_ops.sio",
    "tests/run-pass/octonion_mul_test.sio",
    "tests/run-pass/octonion_associator_gum_validation.sio",
    "tests/run-pass/associator_field_octonion.sio",
    "tests/run-pass/associator_field_pentagon.sio",
    "tests/run-pass/octonion_hessian_fano_annotated.sio",
    "tests/run-pass/knowledge_octonion_inner.sio",
    "tests/run-pass/knowledge_octonion_structure.sio",
    "stdlib/algebra/fano.sio",
    "stdlib/math/octonion.sio",
    "stdlib/onn/g2_activation.sio",
]

CHANNELS = ["real", "e1", "e2", "e3", "e4", "e5", "e6", "e7"]
FANO_TRIPLES = ["123", "145", "176", "246", "257", "347", "365"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--heldout-prompts", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--souc", type=Path, default=REPO_ROOT / "bin/souc")
    parser.add_argument("--max-output-chars", type=int, default=6200)
    parser.add_argument("--no-check", action="store_true")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def heldout_sources(path: Path) -> set[str]:
    sources: set[str] = set()
    if not path.exists():
        return sources
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("source_path"):
            sources.add(str(record["source_path"]))
    return sources


def check_source(souc: Path, source_path: str, timeout: float = 25.0) -> tuple[bool, str]:
    proc = subprocess.run(
        [str(souc), "check", source_path],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    evidence = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, evidence[-1000:]


def descriptor(source_path: str) -> dict:
    lower = source_path.lower()
    if "fano" in lower or "168" in lower:
        target = "Fano triple enumeration and 168 ordered non-collinear triples"
    elif "knowledge" in lower:
        target = "Knowledge<T> octonion variance or inner-product structure"
    elif "associator" in lower:
        target = "octonion associator and non-associative invariant"
    elif "hessian" in lower:
        target = "fano_selective reassociation guard for second derivative"
    else:
        target = "octonion executable algebra invariant"
    return {
        "schema": "sounio.octonion_repair.v1",
        "channels": CHANNELS,
        "fano_triples": FANO_TRIPLES,
        "source_family": Path(source_path).stem,
        "target": target,
        "guardrails": [
            "return one complete Sounio source file",
            "use var for mutable bindings",
            "use &!T for exclusive references",
            "declare IO, Mut, Div, Panic, or Epistemic effects when used",
            "avoid Rust macros and Markdown fences",
        ],
    }


def summary(source: str) -> str:
    lines: list[str] = []
    for raw in source.splitlines():
        s = raw.strip()
        if s.startswith("//") and not s.startswith("//@"):
            lines.append(s.lstrip("/ "))
        if len(" ".join(lines)) > 260:
            break
    return " ".join(lines)[:320] or "compiler-checked octonion/Fano Sounio program"


def make_records(index: int, source_path: str, source: str) -> list[dict]:
    stem = Path(source_path).stem
    desc = descriptor(source_path)
    base = {
        "category": "octonion_repair",
        "source_path": source_path,
        "oracle_kind": "souc_check",
        "oct_descriptor": desc,
        "tags": ["octonion_repair", "compiler_checked", "nonheldout_source"],
    }
    return [
        {
            **base,
            "id": f"octonion_repair_v1_{index:04d}_{stem}_descriptor",
            "instruction": "Generate one complete compiler-checkable Sounio program from this octonion/Fano descriptor.",
            "input": json.dumps(desc, sort_keys=True),
            "output": source,
            "rule": "oct_descriptor_to_program",
        },
        {
            **base,
            "id": f"octonion_repair_v1_{index:04d}_{stem}_repair",
            "instruction": "Repair this broken octonion/Fano Sounio sketch and return one complete source file only.",
            "input": (
                f"Broken sketch family: `{stem}`. Common failures: missing effects, Rust macro syntax, "
                f"bad mutable reference spelling, incomplete Fano loops, and generated Markdown.\n"
                f"Target summary: {summary(source)}\nDescriptor: {json.dumps(desc, sort_keys=True)}"
            ),
            "output": source,
            "rule": "broken_octonion_repair",
        },
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    heldout = heldout_sources(args.heldout_prompts)
    records: list[dict] = []
    skipped: dict[str, str] = {}
    accepted_sources: list[str] = []
    for source_path in SOURCE_CANDIDATES:
        if source_path in heldout:
            skipped[source_path] = "heldout_v2_source_path"
            continue
        path = REPO_ROOT / source_path
        if not path.exists():
            skipped[source_path] = "missing_source"
            continue
        source = read_text(path)
        if len(source) > args.max_output_chars:
            skipped[source_path] = f"output_too_long:{len(source)}"
            continue
        if not args.no_check:
            ok, evidence = check_source(args.souc, source_path)
            if not ok:
                skipped[source_path] = f"souc_check_failed:{evidence}"
                continue
        accepted_sources.append(source_path)
        records.extend(make_records(len(accepted_sources), source_path, source))

    write_jsonl(args.output, records)
    manifest = {
        "schema": "sounio.octonion_repair.v1",
        "rung": "repair_v13i_octonion_repair",
        "record_count": len(records),
        "accepted_source_count": len(accepted_sources),
        "accepted_sources": accepted_sources,
        "heldout_prompt_source_count": len(heldout),
        "category_counts": Counter(row["category"] for row in records),
        "rule_counts": Counter(row["rule"] for row in records),
        "output": str(args.output.relative_to(REPO_ROOT) if args.output.is_relative_to(REPO_ROOT) else args.output),
        "souc_check_filtered": not args.no_check,
        "skipped": skipped,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} records from {len(accepted_sources)} sources -> {args.output}")
    print(f"manifest -> {args.manifest}")
    if skipped:
        print(f"skipped {len(skipped)} candidates")


if __name__ == "__main__":
    main()
