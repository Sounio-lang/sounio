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
WAVE2_VALIDATOR = ROOT_DIR / "scripts" / "research" / "validate_hypercomplex_wave2.py"

REQUIRED_AUDIT_SCOPES = {
    "type-system",
    "ir-metadata",
    "native-lowering",
}
REQUIRED_AUDIT_LANDING_STATES = {
    "mapping-only",
    "landed-validation",
}
REQUIRED_COVERAGE_KINDS = {
    "runtime-counterexample",
    "validation-only",
}


def load_wave2_module():
    spec = importlib.util.spec_from_file_location("hypercomplex_wave2", WAVE2_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load wave2 validator from {WAVE2_VALIDATOR}")
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


def run_fixture(souc_bin: str, fixture_rel: str, expected_stdout: str) -> dict:
    fixture_path = ROOT_DIR / fixture_rel
    env = dict(os.environ)
    env.setdefault("SOUNIO_STDLIB_PATH", str(ROOT_DIR / "stdlib"))
    proc = subprocess.run(
        [souc_bin, "run", str(fixture_path)],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    ok = proc.returncode == 0 and expected_stdout in stdout
    return {
        "fixture": fixture_rel,
        "expected_stdout": expected_stdout,
        "exit_code": proc.returncode,
        "status": "pass" if ok else "fail",
        "stdout": stdout,
        "stderr": stderr,
    }


def validate() -> tuple[dict, int]:
    errors: list[str] = []

    wave2_module = load_wave2_module()
    wave2_report, wave2_code = wave2_module.validate()
    if wave2_code != 0:
        errors.append("wave2 hypercomplex validation failed")

    taxonomy = load_json(DATA_DIR / "taxonomy.v1.json", errors)
    inventory = load_json(DATA_DIR / "inventory.v1.json", errors)
    hazards = load_json(DATA_DIR / "hazards.v1.json", errors)
    semantics = load_json(DATA_DIR / "semantics.v1.json", errors)
    touchpoints = load_json(DATA_DIR / "touchpoints.v1.json", errors)
    expected_fail = load_json(DATA_DIR / "expected_fail.v1.json", errors)
    compiler_audit = load_json(DATA_DIR / "compiler_audit.v1.json", errors)
    readme_path = DATA_DIR / "README.md"
    check(readme_path.exists(), f"missing required file: {readme_path.relative_to(ROOT_DIR)}", errors)

    if errors:
        report = {
            "schema": "sounio.research.hypercomplex.validation_report.v3",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "fail",
            "error_count": len(errors),
            "errors": errors,
            "wave2_status": wave2_report.get("status", "fail"),
        }
        return report, 1

    classes = taxonomy.get("classes", {})
    inventory_entries = inventory.get("entries", [])
    hazard_entries = hazards.get("hazards", [])
    semantic_topics = semantics.get("semantic_topics", [])
    forbidden_laws = semantics.get("forbidden_laws", [])
    touchpoint_entries = touchpoints.get("entries", [])
    expected_fail_entries = expected_fail.get("entries", [])
    audit_entries = compiler_audit.get("entries", [])

    check(expected_fail.get("schema") == "sounio.research.hypercomplex.expected_fail.v1", "unexpected expected_fail schema", errors)
    check(compiler_audit.get("schema") == "sounio.research.hypercomplex.compiler_audit.v1", "unexpected compiler_audit schema", errors)

    allowed_classes = set(classes.keys()) if isinstance(classes, dict) else set()
    inventory_ids = {entry.get("id") for entry in inventory_entries if isinstance(entry.get("id"), str)}
    hazard_ids = {hazard.get("id") for hazard in hazard_entries if isinstance(hazard.get("id"), str)}
    semantic_ids = {entry.get("id") for entry in semantic_topics if isinstance(entry.get("id"), str)}
    touchpoint_ids = {entry.get("id") for entry in touchpoint_entries if isinstance(entry.get("id"), str)}
    forbidden_law_ids = {entry.get("id") for entry in forbidden_laws if isinstance(entry.get("id"), str)}

    check(set(expected_fail.get("coverage_kinds", [])) == REQUIRED_COVERAGE_KINDS, f"coverage_kinds mismatch: expected {sorted(REQUIRED_COVERAGE_KINDS)}", errors)
    check(set(compiler_audit.get("audit_scopes", [])) == REQUIRED_AUDIT_SCOPES, f"audit_scopes mismatch: expected {sorted(REQUIRED_AUDIT_SCOPES)}", errors)
    check(set(compiler_audit.get("landing_states", [])) == REQUIRED_AUDIT_LANDING_STATES, f"landing_states mismatch: expected {sorted(REQUIRED_AUDIT_LANDING_STATES)}", errors)

    coverage_for_law: dict[str, str] = {}
    runtime_entries: list[tuple[str, str]] = []
    expected_fail_counts = {kind: 0 for kind in REQUIRED_COVERAGE_KINDS}
    seen_expected_fail_ids: set[str] = set()
    for entry in expected_fail_entries:
        entry_id = entry.get("id")
        law_id = entry.get("law_id")
        classification = entry.get("classification")
        coverage_kind = entry.get("coverage_kind")
        inventory_list = entry.get("inventory", [])
        touchpoint_list = entry.get("touchpoints", [])
        semantics_list = entry.get("semantics", [])
        hazards_list = entry.get("hazards", [])
        refs = entry.get("references", [])

        check(isinstance(entry_id, str) and entry_id != "", "expected-fail entry missing id", errors)
        if not isinstance(entry_id, str) or entry_id == "":
            continue
        check(entry_id not in seen_expected_fail_ids, f"duplicate expected-fail entry id: {entry_id}", errors)
        seen_expected_fail_ids.add(entry_id)

        check(isinstance(law_id, str) and law_id in forbidden_law_ids, f"expected-fail entry {entry_id} references unknown law_id: {law_id}", errors)
        if isinstance(law_id, str) and law_id in coverage_for_law:
            errors.append(f"forbidden law {law_id} has multiple wave3 coverage entries")
        elif isinstance(law_id, str):
            coverage_for_law[law_id] = entry_id

        check(classification in allowed_classes, f"expected-fail entry {entry_id} has invalid classification: {classification}", errors)
        check(coverage_kind in REQUIRED_COVERAGE_KINDS, f"expected-fail entry {entry_id} has invalid coverage_kind: {coverage_kind}", errors)
        if coverage_kind in expected_fail_counts:
            expected_fail_counts[coverage_kind] += 1
        check(isinstance(entry.get("summary"), str) and entry["summary"] != "", f"expected-fail entry {entry_id} missing summary", errors)
        check(isinstance(inventory_list, list) and inventory_list, f"expected-fail entry {entry_id} must reference inventory ids", errors)
        check(isinstance(touchpoint_list, list) and touchpoint_list, f"expected-fail entry {entry_id} must reference touchpoint ids", errors)
        check(isinstance(semantics_list, list) and semantics_list, f"expected-fail entry {entry_id} must reference semantic topics", errors)
        check(isinstance(hazards_list, list) and hazards_list, f"expected-fail entry {entry_id} must reference hazards", errors)
        check(isinstance(refs, list) and refs, f"expected-fail entry {entry_id} must reference source files", errors)

        for inventory_id in inventory_list if isinstance(inventory_list, list) else []:
            check(inventory_id in inventory_ids, f"expected-fail entry {entry_id} references unknown inventory id: {inventory_id}", errors)
        for touchpoint_id in touchpoint_list if isinstance(touchpoint_list, list) else []:
            check(touchpoint_id in touchpoint_ids, f"expected-fail entry {entry_id} references unknown touchpoint id: {touchpoint_id}", errors)
        for semantic_id in semantics_list if isinstance(semantics_list, list) else []:
            check(semantic_id in semantic_ids, f"expected-fail entry {entry_id} references unknown semantic topic: {semantic_id}", errors)
        for hazard_id in hazards_list if isinstance(hazards_list, list) else []:
            check(hazard_id in hazard_ids, f"expected-fail entry {entry_id} references unknown hazard id: {hazard_id}", errors)
        for ref in refs if isinstance(refs, list) else []:
            check(isinstance(ref, dict), f"expected-fail entry {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors)

        if coverage_kind == "runtime-counterexample":
            fixture = entry.get("fixture")
            check(isinstance(fixture, dict), f"runtime-counterexample entry {entry_id} missing fixture object", errors)
            if isinstance(fixture, dict):
                fixture_path = fixture.get("path")
                expected_stdout = fixture.get("expected_stdout")
                check(isinstance(fixture_path, str) and fixture_path != "", f"runtime-counterexample entry {entry_id} missing fixture path", errors)
                check(isinstance(expected_stdout, str) and expected_stdout != "", f"runtime-counterexample entry {entry_id} missing expected_stdout", errors)
                if isinstance(fixture_path, str) and fixture_path != "":
                    check((ROOT_DIR / fixture_path).exists(), f"runtime-counterexample fixture does not exist: {fixture_path}", errors)
                if isinstance(fixture_path, str) and fixture_path != "" and isinstance(expected_stdout, str) and expected_stdout != "":
                    runtime_entries.append((fixture_path, expected_stdout))
        elif coverage_kind == "validation-only":
            blocking_reason = entry.get("blocking_reason")
            check(isinstance(blocking_reason, str) and blocking_reason != "", f"validation-only entry {entry_id} missing blocking_reason", errors)

    missing_law_coverage = sorted(forbidden_law_ids - set(coverage_for_law.keys()))
    check(not missing_law_coverage, f"missing wave3 expected-fail coverage for laws: {missing_law_coverage}", errors)

    seen_audit_ids: set[str] = set()
    seen_audit_scopes: set[str] = set()
    audit_counts = {cls: 0 for cls in allowed_classes}
    for entry in audit_entries:
        entry_id = entry.get("id")
        audit_scope = entry.get("audit_scope")
        classification = entry.get("classification")
        landing_state = entry.get("landing_state")
        inventory_list = entry.get("inventory", [])
        touchpoint_list = entry.get("touchpoints", [])
        semantics_list = entry.get("semantics", [])
        refs = entry.get("references", [])

        check(isinstance(entry_id, str) and entry_id != "", "compiler audit entry missing id", errors)
        if not isinstance(entry_id, str) or entry_id == "":
            continue
        check(entry_id not in seen_audit_ids, f"duplicate compiler audit entry id: {entry_id}", errors)
        seen_audit_ids.add(entry_id)

        check(audit_scope in REQUIRED_AUDIT_SCOPES, f"compiler audit entry {entry_id} has invalid audit_scope: {audit_scope}", errors)
        if audit_scope in REQUIRED_AUDIT_SCOPES:
            seen_audit_scopes.add(audit_scope)
        check(classification in allowed_classes, f"compiler audit entry {entry_id} has invalid classification: {classification}", errors)
        if classification in audit_counts:
            audit_counts[classification] += 1
        check(landing_state in REQUIRED_AUDIT_LANDING_STATES, f"compiler audit entry {entry_id} has invalid landing_state: {landing_state}", errors)
        check(isinstance(entry.get("summary"), str) and entry["summary"] != "", f"compiler audit entry {entry_id} missing summary", errors)
        check(isinstance(inventory_list, list) and inventory_list, f"compiler audit entry {entry_id} must reference inventory ids", errors)
        check(isinstance(touchpoint_list, list) and touchpoint_list, f"compiler audit entry {entry_id} must reference touchpoint ids", errors)
        check(isinstance(semantics_list, list) and semantics_list, f"compiler audit entry {entry_id} must reference semantic topics", errors)
        check(isinstance(refs, list) and refs, f"compiler audit entry {entry_id} must reference source files", errors)

        for inventory_id in inventory_list if isinstance(inventory_list, list) else []:
            check(inventory_id in inventory_ids, f"compiler audit entry {entry_id} references unknown inventory id: {inventory_id}", errors)
        for touchpoint_id in touchpoint_list if isinstance(touchpoint_list, list) else []:
            check(touchpoint_id in touchpoint_ids, f"compiler audit entry {entry_id} references unknown touchpoint id: {touchpoint_id}", errors)
        for semantic_id in semantics_list if isinstance(semantics_list, list) else []:
            check(semantic_id in semantic_ids, f"compiler audit entry {entry_id} references unknown semantic topic: {semantic_id}", errors)
        for ref in refs if isinstance(refs, list) else []:
            check(isinstance(ref, dict), f"compiler audit entry {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors)

    missing_audit_scopes = sorted(REQUIRED_AUDIT_SCOPES - seen_audit_scopes)
    check(not missing_audit_scopes, f"missing compiler audit coverage for scopes: {missing_audit_scopes}", errors)

    fixture_results: list[dict] = []
    if not errors and runtime_entries:
        souc_bin = resolve_souc(errors)
        if souc_bin:
            for fixture_path, expected_stdout in runtime_entries:
                result = run_fixture(souc_bin, fixture_path, expected_stdout)
                fixture_results.append(result)
                if result["status"] != "pass":
                    errors.append(
                        f"wave3 runtime counterexample failed for {fixture_path}: "
                        f"exit={result['exit_code']} expected_stdout={expected_stdout}"
                    )

    report = {
        "schema": "sounio.research.hypercomplex.validation_report.v3",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass" if not errors else "fail",
        "wave2_status": wave2_report.get("status", "fail"),
        "files": {
            "expected_fail": "research/hypercomplex/expected_fail.v1.json",
            "compiler_audit": "research/hypercomplex/compiler_audit.v1.json",
            "readme": "research/hypercomplex/README.md"
        },
        "counts": {
            "expected_fail_entries": len(expected_fail_entries),
            "runtime_counterexamples": expected_fail_counts["runtime-counterexample"],
            "validation_only_entries": expected_fail_counts["validation-only"],
            "compiler_audit_entries": len(audit_entries),
            "compiler_audit_classifications": audit_counts
        },
        "runtime_fixtures": fixture_results,
        "error_count": len(errors),
        "errors": errors
    }
    return report, 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Wave 3 hypercomplex research manifests.")
    parser.add_argument(
        "--out-json",
        default=str(ROOT_DIR / "artifacts" / "research" / "hypercomplex_wave3_validation.v1.json"),
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
        "hypercomplex-wave3 validation passed: "
        f"{report['counts']['expected_fail_entries']} expected-fail entries, "
        f"{report['counts']['compiler_audit_entries']} compiler audit entries, "
        f"{len(report['runtime_fixtures'])} runtime counterexamples"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
