#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "research" / "hypercomplex"
WAVE3_VALIDATOR = ROOT_DIR / "scripts" / "research" / "validate_hypercomplex_wave3.py"

REQUIRED_SCAFFOLDING_KINDS = {
    "compile-proof-harness",
    "runtime-harness",
    "validation-matrix",
}
REQUIRED_LANDING_STATES = {
    "mapping-only",
    "landed-validation",
}
REQUIRED_WAVE4_TOUCHPOINTS = {
    "wave4_hyper_compile_proof_harness",
    "wave4_parenthesization_fixture_harness",
    "wave4_forbidden_law_validation_matrix",
}


def load_wave3_module():
    spec = importlib.util.spec_from_file_location("hypercomplex_wave3", WAVE3_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load wave3 validator from {WAVE3_VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path, errors: list[str]) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing required file: {path.relative_to(ROOT_DIR)}")
    except json.JSONDecodeError as exc:
        errors.append(f"invalid json in {path.relative_to(ROOT_DIR)}: {exc}")
    return {}


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_reference(ref: dict, errors: list[str]) -> None:
    path_text = ref.get("path")
    check(isinstance(path_text, str) and path_text != "", "reference missing non-empty path", errors)
    if not isinstance(path_text, str) or path_text == "":
        return
    path = ROOT_DIR / path_text
    check(path.exists(), f"referenced path does not exist: {path_text}", errors)
    symbol = ref.get("symbol")
    check(isinstance(symbol, str) and symbol != "", f"reference missing symbol for {path_text}", errors)


def resolve_souc(errors: list[str]) -> str:
    cmd = "source scripts/lib/resolve_souc.sh && sounio_require_souc && printf '%s' \"$SOUC_BIN\""
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown resolve failure"
        errors.append(f"unable to resolve souc via scripts/lib/resolve_souc.sh: {detail}")
        return ""
    souc_bin = proc.stdout.strip()
    if not souc_bin:
        errors.append("souc resolution returned an empty path")
        return ""
    return souc_bin


def compile_fixture(souc_bin: str, fixture_rel: str) -> dict:
    fixture_path = ROOT_DIR / fixture_rel
    out_path = Path("/tmp") / (fixture_path.stem + ".wave4.elf")
    log_path = Path("/tmp") / (fixture_path.stem + ".wave4.log")
    env = dict(os.environ)
    env.setdefault("SOUNIO_STDLIB_PATH", str(ROOT_DIR / "stdlib"))
    proc = subprocess.run(
        [souc_bin, "compile", str(fixture_path), "-o", str(out_path)],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    log_text = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(log_text, encoding="utf-8")
    ok = proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0 and "warning:" not in log_text.lower()
    result = {
        "fixture": fixture_rel,
        "exit_code": proc.returncode,
        "status": "pass" if ok else "fail",
        "output_path": str(out_path),
        "log_path": str(log_path),
    }
    return result


def validate() -> tuple[dict, int]:
    errors: list[str] = []

    wave3_module = load_wave3_module()
    wave3_report, wave3_code = wave3_module.validate()
    if wave3_code != 0:
        errors.append("wave3 hypercomplex validation failed")

    taxonomy = load_json(DATA_DIR / "taxonomy.v1.json", errors)
    semantics = load_json(DATA_DIR / "semantics.v1.json", errors)
    touchpoints = load_json(DATA_DIR / "touchpoints.v1.json", errors)
    hazards = load_json(DATA_DIR / "hazards.v1.json", errors)
    expected_fail = load_json(DATA_DIR / "expected_fail.v1.json", errors)
    compiler_audit = load_json(DATA_DIR / "compiler_audit.v1.json", errors)
    scaffolding = load_json(DATA_DIR / "prototype_scaffolding.v1.json", errors)
    readme_path = DATA_DIR / "README.md"
    check(readme_path.exists(), f"missing required file: {readme_path.relative_to(ROOT_DIR)}", errors)

    if errors:
        report = {
            "schema": "sounio.research.hypercomplex.validation_report.v4",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "fail",
            "error_count": len(errors),
            "errors": errors,
            "wave3_status": wave3_report.get("status", "fail"),
        }
        return report, 1

    classes = taxonomy.get("classes", {})
    touchpoint_entries = touchpoints.get("entries", [])
    semantic_topics = semantics.get("semantic_topics", [])
    prototype_safe_contract = semantics.get("prototype_safe_contract", [])
    hazard_entries = hazards.get("hazards", [])
    expected_fail_entries = expected_fail.get("entries", [])
    compiler_audit_entries = compiler_audit.get("entries", [])
    scaffolding_entries = scaffolding.get("entries", [])

    check(scaffolding.get("schema") == "sounio.research.hypercomplex.prototype_scaffolding.v1", "unexpected prototype_scaffolding schema", errors)
    check(set(scaffolding.get("scaffolding_kinds", [])) == REQUIRED_SCAFFOLDING_KINDS, f"scaffolding_kinds mismatch: expected {sorted(REQUIRED_SCAFFOLDING_KINDS)}", errors)
    check(set(scaffolding.get("landing_states", [])) == REQUIRED_LANDING_STATES, f"landing_states mismatch: expected {sorted(REQUIRED_LANDING_STATES)}", errors)
    check(any("compile-proof" in item for item in prototype_safe_contract if isinstance(item, str)), "prototype_safe_contract must mention compile-proof harness limits", errors)

    allowed_classes = set(classes.keys()) if isinstance(classes, dict) else set()
    touchpoint_ids = {entry.get("id") for entry in touchpoint_entries if isinstance(entry.get("id"), str)}
    semantic_ids = {entry.get("id") for entry in semantic_topics if isinstance(entry.get("id"), str)}
    hazard_ids = {entry.get("id") for entry in hazard_entries if isinstance(entry.get("id"), str)}
    expected_fail_ids = {entry.get("id") for entry in expected_fail_entries if isinstance(entry.get("id"), str)}
    compiler_audit_ids = {entry.get("id") for entry in compiler_audit_entries if isinstance(entry.get("id"), str)}

    missing_wave4_touchpoints = sorted(REQUIRED_WAVE4_TOUCHPOINTS - touchpoint_ids)
    check(not missing_wave4_touchpoints, f"missing Wave 4 touchpoint entries: {missing_wave4_touchpoints}", errors)

    seen_scaffolding_ids: set[str] = set()
    seen_kinds: set[str] = set()
    compile_fixture_paths: list[str] = []
    compile_fixture_count = 0
    for entry in scaffolding_entries:
        entry_id = entry.get("id")
        scaffolding_kind = entry.get("scaffolding_kind")
        classification = entry.get("classification")
        landing_state = entry.get("landing_state")
        inventory_list = entry.get("inventory", [])
        touchpoint_list = entry.get("touchpoints", [])
        semantics_list = entry.get("semantics", [])
        fixtures = entry.get("fixtures", [])
        refs = entry.get("references", [])

        check(isinstance(entry_id, str) and entry_id != "", "prototype scaffolding entry missing id", errors)
        if not isinstance(entry_id, str) or entry_id == "":
            continue
        check(entry_id not in seen_scaffolding_ids, f"duplicate prototype scaffolding id: {entry_id}", errors)
        seen_scaffolding_ids.add(entry_id)

        check(scaffolding_kind in REQUIRED_SCAFFOLDING_KINDS, f"prototype scaffolding {entry_id} has invalid kind: {scaffolding_kind}", errors)
        if scaffolding_kind in REQUIRED_SCAFFOLDING_KINDS:
            seen_kinds.add(scaffolding_kind)
        check(classification in allowed_classes, f"prototype scaffolding {entry_id} has invalid classification: {classification}", errors)
        check(landing_state in REQUIRED_LANDING_STATES, f"prototype scaffolding {entry_id} has invalid landing_state: {landing_state}", errors)
        check(isinstance(entry.get("summary"), str) and entry["summary"] != "", f"prototype scaffolding {entry_id} missing summary", errors)
        check(isinstance(inventory_list, list) and inventory_list, f"prototype scaffolding {entry_id} must reference inventory ids", errors)
        check(isinstance(touchpoint_list, list) and touchpoint_list, f"prototype scaffolding {entry_id} must reference touchpoint ids", errors)
        check(isinstance(semantics_list, list) and semantics_list, f"prototype scaffolding {entry_id} must reference semantic topics", errors)
        check(isinstance(fixtures, list) and fixtures, f"prototype scaffolding {entry_id} must reference fixtures", errors)
        check(isinstance(refs, list) and refs, f"prototype scaffolding {entry_id} must reference source files", errors)

        for touchpoint_id in touchpoint_list if isinstance(touchpoint_list, list) else []:
            check(touchpoint_id in touchpoint_ids, f"prototype scaffolding {entry_id} references unknown touchpoint id: {touchpoint_id}", errors)
        for semantic_id in semantics_list if isinstance(semantics_list, list) else []:
            check(semantic_id in semantic_ids, f"prototype scaffolding {entry_id} references unknown semantic topic: {semantic_id}", errors)
        for ref in refs if isinstance(refs, list) else []:
            check(isinstance(ref, dict), f"prototype scaffolding {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors)

        for fixture in fixtures if isinstance(fixtures, list) else []:
            check(isinstance(fixture, dict), f"prototype scaffolding {entry_id} has malformed fixture entry", errors)
            if not isinstance(fixture, dict):
                continue
            fixture_path = fixture.get("path")
            fixture_mode = fixture.get("mode")
            check(isinstance(fixture_path, str) and fixture_path != "", f"prototype scaffolding {entry_id} fixture missing path", errors)
            if isinstance(fixture_path, str) and fixture_path != "":
                check((ROOT_DIR / fixture_path).exists(), f"prototype scaffolding fixture does not exist: {fixture_path}", errors)
            check(isinstance(fixture_mode, str) and fixture_mode != "", f"prototype scaffolding {entry_id} fixture missing mode", errors)
            if scaffolding_kind == "compile-proof-harness" and fixture_mode == "compile-pass" and isinstance(fixture_path, str) and fixture_path != "":
                compile_fixture_paths.append(fixture_path)
                compile_fixture_count += 1
            if scaffolding_kind == "validation-matrix" and isinstance(fixture_path, str) and fixture_path.endswith("expected_fail.v1.json"):
                check("wave4_forbidden_law_validation_matrix" in touchpoint_list, "validation matrix must link the Wave 4 forbidden-law touchpoint", errors)

    missing_scaffolding_kinds = sorted(REQUIRED_SCAFFOLDING_KINDS - seen_kinds)
    check(not missing_scaffolding_kinds, f"missing prototype scaffolding coverage for kinds: {missing_scaffolding_kinds}", errors)
    check(compile_fixture_count >= 3, "Wave 4 requires at least 3 compile-proof fixtures", errors)

    hazard_refs = json.dumps(hazards)
    check("wave3_flatten_nonassoc_counterexample.sio" in hazard_refs, "hazards map must reference the Wave 3 flatten counterexample", errors)
    check("wave3_cancel_nonzero_sedenion_factors_counterexample.sio" in hazard_refs, "hazards map must reference the Wave 3 zero-divisor counterexample", errors)
    check("wave3_sedenion_norm_identity_counterexample.sio" in hazard_refs, "hazards map must reference the Wave 3 norm counterexample", errors)

    fixture_results: list[dict] = []
    if not errors and compile_fixture_paths:
        souc_bin = resolve_souc(errors)
        if souc_bin:
            for fixture_path in compile_fixture_paths:
                result = compile_fixture(souc_bin, fixture_path)
                fixture_results.append(result)
                if result["status"] != "pass":
                    errors.append(f"Wave 4 compile-proof fixture failed: {fixture_path}")

    report = {
        "schema": "sounio.research.hypercomplex.validation_report.v4",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass" if not errors else "fail",
        "wave3_status": wave3_report.get("status", "fail"),
        "files": {
            "prototype_scaffolding": "research/hypercomplex/prototype_scaffolding.v1.json",
            "semantics": "research/hypercomplex/semantics.v1.json",
            "touchpoints": "research/hypercomplex/touchpoints.v1.json",
            "hazards": "research/hypercomplex/hazards.v1.json",
            "readme": "research/hypercomplex/README.md"
        },
        "counts": {
            "prototype_scaffolding_entries": len(scaffolding_entries),
            "compile_fixture_count": compile_fixture_count,
            "touchpoint_entries": len(touchpoint_entries),
            "expected_fail_entries": len(expected_fail_entries),
            "compiler_audit_entries": len(compiler_audit_entries)
        },
        "compile_fixtures": fixture_results,
        "error_count": len(errors),
        "errors": errors
    }
    return report, 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Wave 4 hypercomplex research scaffolding.")
    parser.add_argument(
        "--out-json",
        default=str(ROOT_DIR / "artifacts" / "research" / "hypercomplex_wave4_validation.v1.json"),
        help="write validation report JSON to this path",
    )
    args = parser.parse_args()

    report, code = validate()
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if code != 0:
        for error in report["errors"]:
            print(f"error: {error}", file=sys.stderr)
        return code

    print(
        "hypercomplex-wave4 validation passed: "
        f"{report['counts']['prototype_scaffolding_entries']} scaffolding entries, "
        f"{report['counts']['compile_fixture_count']} compile fixtures"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
