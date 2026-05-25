#!/usr/bin/env python3
"""Build the v13 octonion/O-SSM structure-hybrid training dataset.

The builder keeps the rung intentionally small: every output is a complete
repo source file that passes `souc check`, and Mandelbrot/Hessian prompts are
left for the next rung.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "datasets/sounio-ai-structure-hybrid/structure_hybrid.v1.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "datasets/sounio-ai-structure-hybrid/manifest.v1.json"

FANO_TRIPLES = ["123", "145", "176", "246", "257", "347", "365"]
CHANNELS = ["real", "e1", "e2", "e3", "e4", "e5", "e6", "e7"]

SOURCE_CANDIDATES = [
    # O-SSM control
    "examples/conversational_ossm/associator_telemetry.sio",
    "examples/conversational_ossm/bidirectional_ossm_v0.sio",
    "examples/conversational_ossm/o_ssm_conflict.sio",
    "tests/run-pass/ossm_knowledge_basic.sio",
    "tests/run-pass/gpu_epistemic_f64_ossm_parity.sio",
    "examples/cognitive_ossm/cognitive_ossm.sio",
    "examples/ossm_knowledge.sio",
    "examples/ossm_selective.sio",
    "examples/ossm_fano_selective.sio",
    # Octonion/algebra invariants
    "tests/native-v2/science_spine/octonion_fano_168.sio",
    "tests/native-v2/science_spine/associator_gum.sio",
    "tests/native-v2/science_spine/q_deform_octonion.sio",
    "tests/run-pass/octonion_168_test.sio",
    "tests/run-pass/fano_basics.sio",
    "tests/run-pass/octonion_basic_ops.sio",
    "tests/run-pass/octonion_basic_ops_standalone.sio",
    "tests/run-pass/octonion_mul_test.sio",
    "tests/run-pass/octonion_cayley_dickson.sio",
    "tests/run-pass/octonion_associator_gum_validation.sio",
    "tests/run-pass/associator_field_octonion.sio",
    "tests/run-pass/associator_variance_mc.sio",
    "tests/run-pass/associator_field_pentagon.sio",
    # Knowledge/science spine
    "tests/native-v2/science_spine/knowledge_octonion_variance.sio",
    "tests/run-pass/knowledge_octonion_inner.sio",
    "tests/run-pass/knowledge_octonion_structure.sio",
]

SCIENCE_SPINE_STDOUTS = {
    "tests/native-v2/science_spine/octonion_fano_168.sio": "tests/native-v2/science_spine/octonion_fano_168.stdout",
    "tests/native-v2/science_spine/associator_gum.sio": "tests/native-v2/science_spine/associator_gum.stdout",
    "tests/native-v2/science_spine/q_deform_octonion.sio": "tests/native-v2/science_spine/q_deform_octonion.stdout",
    "tests/native-v2/science_spine/knowledge_octonion_variance.sio": "tests/native-v2/science_spine/knowledge_octonion_variance.stdout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--souc", type=Path, default=REPO_ROOT / "bin/souc")
    parser.add_argument("--max-output-chars", type=int, default=6000)
    parser.add_argument("--no-check", action="store_true", help="Skip souc check filtering.")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def source_summary(text: str) -> str:
    comments = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") and not stripped.startswith("//@"):
            comments.append(stripped.lstrip("/ "))
        if len(" ".join(comments)) > 220:
            break
    return " ".join(comments)[:260] or "compiler-backed Sounio source"


def declared_expected_stdout(source_path: str) -> str | None:
    stdout_path = SCIENCE_SPINE_STDOUTS.get(source_path)
    if stdout_path:
        path = REPO_ROOT / stdout_path
        if path.exists():
            return read_text(path).strip()
    text = read_text(REPO_ROOT / source_path)
    for line in text.splitlines():
        marker = "//@ expect-stdout:"
        if line.strip().startswith(marker):
            return line.split(marker, 1)[1].strip()
    return None


def categorize(source_path: str) -> str:
    name = source_path.lower()
    if "ossm" in name or "o_ssm" in name or "conversational_ossm" in name:
        return "ossm_control"
    if "knowledge_octonion" in name:
        return "knowledge_octonion"
    if "science_spine" in name:
        return "science_spine"
    return "octonion_invariant"


def primary_rule(category: str, source_path: str) -> str:
    if category == "ossm_control":
        return "ossm_state_to_program"
    if category in {"octonion_invariant", "knowledge_octonion"}:
        return "oct_descriptor_to_program"
    if "associator" in source_path:
        return "broken_to_repaired_invariant"
    return "nl_to_executable_invariant"


def descriptor_for(source_path: str, category: str, expected: str | None) -> dict:
    lower = source_path.lower()
    if category == "ossm_control":
        target = "state/action control shape"
    elif "168" in lower:
        target = "168 ordered distinct imaginary triples"
    elif "norm" in lower or "cayley" in lower:
        target = "norm composition"
    elif "associator" in lower:
        target = "non-associative associator signal"
    elif "knowledge" in lower:
        target = "Knowledge<T> octonion variance/inner structure"
    elif "fano" in lower:
        target = "Fano line multiplication orientation"
    else:
        target = "executable science/control invariant"
    return {
        "schema": "sounio.octo_ossm.structure_hybrid.v1",
        "channels": CHANNELS,
        "fano_triples": FANO_TRIPLES,
        "category": category,
        "target": target,
        "expected_invariant": expected or "souc check succeeds",
    }


def check_source(souc: Path, source_path: str, timeout: float = 20.0) -> tuple[bool, str]:
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


def validated_expected_stdout(souc: Path, source_path: str, timeout: float = 20.0) -> tuple[str | None, str | None]:
    declared = declared_expected_stdout(source_path)
    if declared is None:
        return None, None
    proc = subprocess.run(
        [str(souc), "run", source_path],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        evidence = (proc.stdout + proc.stderr).strip()
        return None, f"souc_run_failed: {evidence[-1000:]}"
    return proc.stdout.strip(), None


def make_record(index: int, source_path: str, source: str, expected: str | None) -> dict:
    category = categorize(source_path)
    rule = primary_rule(category, source_path)
    descriptor = descriptor_for(source_path, category, expected)
    stem = Path(source_path).stem
    if rule == "ossm_state_to_program":
        instruction = "Write a compiler-checkable Sounio control program from an O-SSM state/action description."
        input_text = (
            f"O-SSM target: preserve explicit state/action/control-flow shape for `{stem}`.\n"
            f"Descriptor: {json.dumps(descriptor, sort_keys=True)}"
        )
    elif rule == "broken_to_repaired_invariant":
        instruction = "Repair this scientific Sounio invariant program and return one complete source file only."
        input_text = (
            f"Broken sketch: an associator/invariant example for `{stem}` has Rust-style fences and unstable syntax.\n"
            f"Repair target descriptor: {json.dumps(descriptor, sort_keys=True)}"
        )
    elif rule == "oct_descriptor_to_program":
        instruction = "Generate one complete Sounio program from this octonion/Fano descriptor."
        input_text = json.dumps(descriptor, sort_keys=True)
    else:
        instruction = "Write one complete executable Sounio invariant program for this scientific description."
        input_text = f"{source_summary(source)}\nDescriptor: {json.dumps(descriptor, sort_keys=True)}"
    return {
        "id": f"structure_hybrid_v1_{index:04d}_{stem}",
        "instruction": instruction,
        "input": input_text,
        "output": source,
        "category": category,
        "rule": rule,
        "source_path": source_path,
        "oracle_kind": "expect_stdout" if expected else "souc_check",
        "expected_stdout": expected,
        "oct_descriptor": descriptor,
        "tags": ["structure_hybrid", category, rule, "compiler_checked"],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    records: list[dict] = []
    skipped: dict[str, str] = {}
    for source_path in SOURCE_CANDIDATES:
        if "mandelbrot" in source_path.lower():
            skipped[source_path] = "mandelbrot_reserved_for_v14"
            continue
        path = REPO_ROOT / source_path
        if not path.exists():
            skipped[source_path] = "missing_source"
            continue
        source = read_text(path)
        if len(source) > args.max_output_chars:
            skipped[source_path] = "output_too_long"
            continue
        if not args.no_check:
            ok, evidence = check_source(args.souc, source_path)
            if not ok:
                skipped[source_path] = f"souc_check_failed: {evidence}"
                continue
        expected, run_error = validated_expected_stdout(args.souc, source_path)
        if run_error is not None:
            skipped[source_path] = run_error
            continue
        records.append(make_record(len(records) + 1, source_path, source, expected))

    write_jsonl(args.output, records)
    manifest = {
        "schema": "sounio.octo_ossm.structure_hybrid.v1",
        "rung": "repair_v13_octo_ossm_hybrid",
        "record_count": len(records),
        "category_counts": Counter(row["category"] for row in records),
        "rule_counts": Counter(row["rule"] for row in records),
        "quotas_target": {
            "octonion_algebra": 0.40,
            "ossm_control": 0.30,
            "knowledge_science_spine": 0.20,
            "syntax_runtime_guard": 0.10,
        },
        "output": str(args.output.relative_to(REPO_ROOT) if args.output.is_relative_to(REPO_ROOT) else args.output),
        "souc_check_filtered": not args.no_check,
        "expected_stdout_source": "souc_run_current_output_when_declared",
        "mandelbrot_d2": "excluded_for_v14",
        "skipped": skipped,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} records -> {args.output}")
    print(f"manifest -> {args.manifest}")
    if skipped:
        print(f"skipped {len(skipped)} candidates")


if __name__ == "__main__":
    main()
