#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "research" / "hypercomplex"
WAVE9_VALIDATOR = ROOT_DIR / "scripts" / "research" / "validate_hypercomplex_wave9.py"
WAVE10_DIFF_SELFTEST = ROOT_DIR / "scripts" / "research" / "hypercomplex_wave10_diff_selftest.sh"

REQUIRED_WAVE10_TOUCHPOINTS = {
    "wave10_law_profile_audit_consumer_harness",
}
REQUIRED_WAVE10_AUDITS = {
    "ir.hyper_expr_law_profile_audit_consumer",
}
REQUIRED_WAVE10_SCAFFOLDING = {
    "wave10_law_profile_audit_runtime_harness",
}


def load_wave9_module():
    spec = importlib.util.spec_from_file_location("hypercomplex_wave9", WAVE9_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load wave9 validator from {WAVE9_VALIDATOR}")
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


def validate_reference(ref: dict, errors: list[str], require_symbol_match: bool = False) -> None:
    path_text = ref.get("path")
    check(isinstance(path_text, str) and path_text != "", "reference missing non-empty path", errors)
    if not isinstance(path_text, str) or path_text == "":
        return
    path = ROOT_DIR / path_text
    check(path.exists(), f"referenced path does not exist: {path_text}", errors)
    symbol = ref.get("symbol")
    check(isinstance(symbol, str) and symbol != "", f"reference missing symbol for {path_text}", errors)
    if not require_symbol_match or not path.exists() or not isinstance(symbol, str) or symbol == "":
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"cannot symbol-check non-text reference: {path_text}")
        return
    check(symbol in text, f"reference symbol not found in {path_text}: {symbol}", errors)


def run_baseline_comparison(script_path: Path) -> dict:
    log_path = Path("/tmp") / "hypercomplex_wave10_diff_selftest.validation.log"
    proc = subprocess.run(
        ["bash", str(script_path)],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    log_text = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(log_text, encoding="utf-8")
    ok = proc.returncode == 0 and "baseline comparison passed" in log_text
    return {
        "script": str(script_path.relative_to(ROOT_DIR)),
        "exit_code": proc.returncode,
        "status": "pass" if ok else "fail",
        "log_path": str(log_path),
    }


def validate() -> tuple[dict, int]:
    errors: list[str] = []

    wave9_module = load_wave9_module()
    wave9_report, wave9_code = wave9_module.validate()
    if wave9_code != 0:
        errors.append("wave9 hypercomplex validation failed")

    inventory = load_json(DATA_DIR / "inventory.v1.json", errors)
    semantics = load_json(DATA_DIR / "semantics.v1.json", errors)
    touchpoints = load_json(DATA_DIR / "touchpoints.v1.json", errors)
    compiler_audit = load_json(DATA_DIR / "compiler_audit.v1.json", errors)
    scaffolding = load_json(DATA_DIR / "prototype_scaffolding.v1.json", errors)

    readme_path = DATA_DIR / "README.md"
    fixture_source_path = ROOT_DIR / "tests" / "research" / "hypercomplex" / "wave10_hyper_expr_law_profile_audit_compile_proof.sio"
    check(readme_path.exists(), f"missing required file: {readme_path.relative_to(ROOT_DIR)}", errors)
    check(fixture_source_path.exists(), f"missing wave10 compile-proof fixture: {fixture_source_path.relative_to(ROOT_DIR)}", errors)
    check(WAVE10_DIFF_SELFTEST.exists(), f"missing wave10 differential self-test script: {WAVE10_DIFF_SELFTEST.relative_to(ROOT_DIR)}", errors)

    if errors:
        report = {
            "schema": "sounio.research.hypercomplex.validation_report.v10",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "fail",
            "error_count": len(errors),
            "errors": errors,
            "wave9_status": wave9_report.get("status", "fail"),
        }
        return report, 1

    inventory_entries = inventory.get("entries", [])
    inventory_by_id = {
        entry.get("id"): entry for entry in inventory_entries if isinstance(entry.get("id"), str)
    }
    touchpoint_entries = touchpoints.get("entries", [])
    touchpoint_by_id = {
        entry.get("id"): entry for entry in touchpoint_entries if isinstance(entry.get("id"), str)
    }
    audit_entries = compiler_audit.get("entries", [])
    audit_by_id = {
        entry.get("id"): entry for entry in audit_entries if isinstance(entry.get("id"), str)
    }
    scaffolding_entries = scaffolding.get("entries", [])
    scaffolding_by_id = {
        entry.get("id"): entry for entry in scaffolding_entries if isinstance(entry.get("id"), str)
    }

    prototype_safe_contract = semantics.get("prototype_safe_contract", [])
    check(
        any("internal audit tags derived from law-profile metadata" in item for item in prototype_safe_contract if isinstance(item, str)),
        "prototype_safe_contract must mention internal audit tags",
        errors,
    )

    missing_touchpoints = sorted(REQUIRED_WAVE10_TOUCHPOINTS - set(touchpoint_by_id.keys()))
    check(not missing_touchpoints, f"missing required wave10 touchpoint entries: {missing_touchpoints}", errors)
    missing_audits = sorted(REQUIRED_WAVE10_AUDITS - set(audit_by_id.keys()))
    check(not missing_audits, f"missing required wave10 compiler audit entries: {missing_audits}", errors)
    missing_scaffolding = sorted(REQUIRED_WAVE10_SCAFFOLDING - set(scaffolding_by_id.keys()))
    check(not missing_scaffolding, f"missing required wave10 scaffolding entries: {missing_scaffolding}", errors)

    for entry_id in REQUIRED_WAVE10_TOUCHPOINTS:
        entry = touchpoint_by_id.get(entry_id, {})
        check(entry.get("classification") == "prototype-safe", f"touchpoint {entry_id} must be prototype-safe", errors)
        check(entry.get("landing_state") == "landed-validation", f"touchpoint {entry_id} must be landed-validation", errors)
        for ref in entry.get("references", []) if isinstance(entry.get("references"), list) else []:
            check(isinstance(ref, dict), f"touchpoint {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors, require_symbol_match=True)

    for entry_id in REQUIRED_WAVE10_AUDITS:
        entry = audit_by_id.get(entry_id, {})
        check(entry.get("classification") == "prototype-safe", f"compiler audit {entry_id} must be prototype-safe", errors)
        check(entry.get("landing_state") == "landed-validation", f"compiler audit {entry_id} must be landed-validation", errors)
        for ref in entry.get("references", []) if isinstance(entry.get("references"), list) else []:
            check(isinstance(ref, dict), f"compiler audit {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors, require_symbol_match=True)

    for entry_id in REQUIRED_WAVE10_SCAFFOLDING:
        entry = scaffolding_by_id.get(entry_id, {})
        check(entry.get("classification") == "prototype-safe", f"prototype scaffolding {entry_id} must be prototype-safe", errors)
        check(entry.get("landing_state") == "landed-validation", f"prototype scaffolding {entry_id} must be landed-validation", errors)
        fixtures = entry.get("fixtures", []) if isinstance(entry, dict) else []
        check(isinstance(fixtures, list) and fixtures, f"prototype scaffolding {entry_id} must reference comparison fixtures", errors)
        for fixture in fixtures if isinstance(fixtures, list) else []:
            check(isinstance(fixture, dict), f"prototype scaffolding fixture malformed for {entry_id}", errors)
            if not isinstance(fixture, dict):
                continue
            fixture_path = fixture.get("path")
            fixture_mode = fixture.get("mode")
            check(isinstance(fixture_path, str) and fixture_path != "", f"prototype scaffolding fixture missing path for {entry_id}", errors)
            if isinstance(fixture_path, str) and fixture_path != "":
                check((ROOT_DIR / fixture_path).exists(), f"prototype scaffolding fixture does not exist: {fixture_path}", errors)
            check(fixture_mode == "compile-pass", f"prototype scaffolding fixture must be compile-pass: {fixture_path}", errors)
        for ref in entry.get("references", []) if isinstance(entry.get("references"), list) else []:
            check(isinstance(ref, dict), f"prototype scaffolding {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors, require_symbol_match=True)

    inventory_entry = inventory_by_id.get("ir.hyper_metadata_audit_consumer", {})
    check(
        "internal audit tag" in inventory_entry.get("summary", ""),
        "ir.hyper_metadata_audit_consumer summary must mention internal audit tags",
        errors,
    )

    diff_report = run_baseline_comparison(WAVE10_DIFF_SELFTEST)
    if diff_report["status"] != "pass":
        errors.append("wave10 baseline comparison failed")

    if errors:
        report = {
            "schema": "sounio.research.hypercomplex.validation_report.v10",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "fail",
            "error_count": len(errors),
            "errors": errors,
            "wave9_status": wave9_report.get("status", "fail"),
            "baseline_comparison": diff_report,
        }
        return report, 1

    report = {
        "schema": "sounio.research.hypercomplex.validation_report.v10",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass",
        "wave9_status": wave9_report.get("status", "pass"),
        "baseline_comparison": diff_report,
        "inventory_entry": inventory_entry.get("id"),
    }
    return report, 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate hypercomplex wave 10")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    report, code = validate()
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
