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
WAVE4_VALIDATOR = ROOT_DIR / "scripts" / "research" / "validate_hypercomplex_wave4.py"

REQUIRED_WAVE5_TOUCHPOINTS = {
    "wave5_reassociate_metadata_compile_proof_harness",
}
REQUIRED_WAVE5_AUDITS = {
    "ir.reassociate_strategy_compile_proof_audit",
}
REQUIRED_VALIDATION_CONTRACT_KEYS = {
    "required_inventory",
    "required_touchpoints",
    "required_semantics",
    "required_hazards",
    "forbidden_scaffolding_touchpoints",
    "notes",
}


def load_wave4_module():
    spec = importlib.util.spec_from_file_location("hypercomplex_wave4", WAVE4_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load wave4 validator from {WAVE4_VALIDATOR}")
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
    out_path = Path("/tmp") / (fixture_path.stem + ".wave5.elf")
    log_path = Path("/tmp") / (fixture_path.stem + ".wave5.log")
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
    return {
        "fixture": fixture_rel,
        "exit_code": proc.returncode,
        "status": "pass" if ok else "fail",
        "output_path": str(out_path),
        "log_path": str(log_path),
    }


def validate() -> tuple[dict, int]:
    errors: list[str] = []

    wave4_module = load_wave4_module()
    wave4_report, wave4_code = wave4_module.validate()
    if wave4_code != 0:
        errors.append("wave4 hypercomplex validation failed")

    taxonomy = load_json(DATA_DIR / "taxonomy.v1.json", errors)
    inventory = load_json(DATA_DIR / "inventory.v1.json", errors)
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
            "schema": "sounio.research.hypercomplex.validation_report.v5",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "fail",
            "error_count": len(errors),
            "errors": errors,
            "wave4_status": wave4_report.get("status", "fail"),
        }
        return report, 1

    classes = taxonomy.get("classes", {})
    allowed_classes = set(classes.keys()) if isinstance(classes, dict) else set()
    inventory_ids = {entry.get("id") for entry in inventory.get("entries", []) if isinstance(entry.get("id"), str)}
    semantic_topics = semantics.get("semantic_topics", [])
    semantic_ids = {entry.get("id") for entry in semantic_topics if isinstance(entry.get("id"), str)}
    touchpoint_entries = touchpoints.get("entries", [])
    touchpoint_by_id = {
        entry.get("id"): entry for entry in touchpoint_entries if isinstance(entry.get("id"), str)
    }
    touchpoint_ids = set(touchpoint_by_id.keys())
    hazard_entries = hazards.get("hazards", [])
    hazard_by_id = {
        entry.get("id"): entry for entry in hazard_entries if isinstance(entry.get("id"), str)
    }
    hazard_ids = set(hazard_by_id.keys())
    expected_fail_entries = expected_fail.get("entries", [])
    expected_fail_by_id = {
        entry.get("id"): entry for entry in expected_fail_entries if isinstance(entry.get("id"), str)
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
    check(any("validation-only" in item for item in prototype_safe_contract if isinstance(item, str)), "prototype_safe_contract must mention validation-only forbidden-law gaps", errors)

    missing_touchpoints = sorted(REQUIRED_WAVE5_TOUCHPOINTS - touchpoint_ids)
    check(not missing_touchpoints, f"missing Wave 5 touchpoint entries: {missing_touchpoints}", errors)
    missing_audits = sorted(REQUIRED_WAVE5_AUDITS - set(audit_by_id.keys()))
    check(not missing_audits, f"missing Wave 5 compiler audit entries: {missing_audits}", errors)
    check("wave5_reassociate_metadata_compile_proof_harness" in scaffolding_by_id, "missing Wave 5 prototype scaffolding entry", errors)

    for entry_id in REQUIRED_WAVE5_TOUCHPOINTS:
        entry = touchpoint_by_id.get(entry_id, {})
        check(entry.get("classification") == "prototype-safe", f"Wave 5 touchpoint {entry_id} must be prototype-safe", errors)
        check(entry.get("landing_state") == "landed-validation", f"Wave 5 touchpoint {entry_id} must be landed-validation", errors)

    for entry_id in REQUIRED_WAVE5_AUDITS:
        entry = audit_by_id.get(entry_id, {})
        check(entry.get("classification") == "prototype-safe", f"Wave 5 compiler audit {entry_id} must be prototype-safe", errors)
        check(entry.get("landing_state") == "landed-validation", f"Wave 5 compiler audit {entry_id} must be landed-validation", errors)
        for ref in entry.get("references", []) if isinstance(entry.get("references"), list) else []:
            check(isinstance(ref, dict), f"Wave 5 compiler audit {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors)

    gap_entry = expected_fail_by_id.get("generic_distribute_or_factor.contract_gap")
    check(isinstance(gap_entry, dict), "missing generic_distribute_or_factor.contract_gap entry", errors)
    validation_contract = gap_entry.get("validation_contract") if isinstance(gap_entry, dict) else None
    check(isinstance(validation_contract, dict), "generic_distribute_or_factor.contract_gap must include validation_contract", errors)

    if isinstance(validation_contract, dict):
        missing_keys = sorted(REQUIRED_VALIDATION_CONTRACT_KEYS - set(validation_contract.keys()))
        check(not missing_keys, f"generic_distribute_or_factor.contract_gap missing validation_contract keys: {missing_keys}", errors)

        required_inventory = validation_contract.get("required_inventory", [])
        required_touchpoints = validation_contract.get("required_touchpoints", [])
        required_semantics = validation_contract.get("required_semantics", [])
        required_hazards = validation_contract.get("required_hazards", [])
        forbidden_scaffolding_touchpoints = validation_contract.get("forbidden_scaffolding_touchpoints", [])
        notes = validation_contract.get("notes")

        check(isinstance(notes, str) and notes != "", "validation_contract notes must be non-empty", errors)
        for inventory_id in required_inventory if isinstance(required_inventory, list) else []:
            check(inventory_id in inventory_ids, f"validation_contract references unknown inventory id: {inventory_id}", errors)
        for semantic_id in required_semantics if isinstance(required_semantics, list) else []:
            check(semantic_id in semantic_ids, f"validation_contract references unknown semantic id: {semantic_id}", errors)
        for hazard_id in required_hazards if isinstance(required_hazards, list) else []:
            check(hazard_id in hazard_ids, f"validation_contract references unknown hazard id: {hazard_id}", errors)
        for touchpoint_id in required_touchpoints if isinstance(required_touchpoints, list) else []:
            check(touchpoint_id in touchpoint_ids, f"validation_contract references unknown touchpoint id: {touchpoint_id}", errors)
            if touchpoint_id in touchpoint_by_id:
                touchpoint = touchpoint_by_id[touchpoint_id]
                check(touchpoint.get("classification") == "research-only", f"validation-only touchpoint {touchpoint_id} must stay research-only", errors)
                check(touchpoint.get("landing_state") == "deferred", f"validation-only touchpoint {touchpoint_id} must stay deferred", errors)

        for entry in scaffolding_entries:
            if not isinstance(entry, dict):
                continue
            touchpoints_list = entry.get("touchpoints", [])
            for forbidden_touchpoint in forbidden_scaffolding_touchpoints if isinstance(forbidden_scaffolding_touchpoints, list) else []:
                check(forbidden_touchpoint not in touchpoints_list, f"prototype scaffolding entry {entry.get('id')} must not land against forbidden touchpoint {forbidden_touchpoint}", errors)

    distributivity_hazard = hazard_by_id.get("distributivity_sensitive_rewrites", {})
    hazard_refs = distributivity_hazard.get("references", []) if isinstance(distributivity_hazard, dict) else []
    check(any(isinstance(ref, dict) and ref.get("path") == "research/hypercomplex/expected_fail.v1.json" for ref in hazard_refs), "distributivity_sensitive_rewrites must reference expected_fail.v1.json", errors)

    wave5_scaffold = scaffolding_by_id.get("wave5_reassociate_metadata_compile_proof_harness", {})
    compile_fixture_paths: list[str] = []
    fixtures = wave5_scaffold.get("fixtures", []) if isinstance(wave5_scaffold, dict) else []
    check(isinstance(fixtures, list) and fixtures, "Wave 5 scaffolding entry must reference compile fixtures", errors)
    for fixture in fixtures if isinstance(fixtures, list) else []:
        check(isinstance(fixture, dict), "Wave 5 scaffolding fixture must be an object", errors)
        if not isinstance(fixture, dict):
            continue
        fixture_path = fixture.get("path")
        fixture_mode = fixture.get("mode")
        check(isinstance(fixture_path, str) and fixture_path != "", "Wave 5 scaffolding fixture missing path", errors)
        if isinstance(fixture_path, str) and fixture_path != "":
            check((ROOT_DIR / fixture_path).exists(), f"Wave 5 scaffolding fixture does not exist: {fixture_path}", errors)
        check(fixture_mode == "compile-pass", f"Wave 5 scaffolding fixture must be compile-pass: {fixture_path}", errors)
        if isinstance(fixture_path, str) and fixture_path != "" and fixture_mode == "compile-pass":
            compile_fixture_paths.append(fixture_path)

    fixture_results: list[dict] = []
    if not errors and compile_fixture_paths:
        souc_bin = resolve_souc(errors)
        if souc_bin:
            for fixture_path in compile_fixture_paths:
                result = compile_fixture(souc_bin, fixture_path)
                fixture_results.append(result)
                if result["status"] != "pass":
                    errors.append(f"Wave 5 compile-proof fixture failed: {fixture_path}")

    readme_text = readme_path.read_text(encoding="utf-8")
    check("hypercomplex_wave5_gate.sh" in readme_text, "README must mention hypercomplex_wave5_gate.sh", errors)
    check("metadata-carriage" in readme_text, "README must describe Wave 5 metadata-carriage scaffolding", errors)

    report = {
        "schema": "sounio.research.hypercomplex.validation_report.v5",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass" if not errors else "fail",
        "wave4_status": wave4_report.get("status", "fail"),
        "files": {
            "prototype_scaffolding": "research/hypercomplex/prototype_scaffolding.v1.json",
            "expected_fail": "research/hypercomplex/expected_fail.v1.json",
            "compiler_audit": "research/hypercomplex/compiler_audit.v1.json",
            "touchpoints": "research/hypercomplex/touchpoints.v1.json",
            "semantics": "research/hypercomplex/semantics.v1.json",
            "readme": "research/hypercomplex/README.md",
        },
        "counts": {
            "wave5_touchpoints": len(REQUIRED_WAVE5_TOUCHPOINTS),
            "wave5_audits": len(REQUIRED_WAVE5_AUDITS),
            "compile_fixture_count": len(compile_fixture_paths),
            "validation_only_entries": sum(1 for entry in expected_fail_entries if entry.get("coverage_kind") == "validation-only"),
        },
        "compile_fixtures": fixture_results,
        "error_count": len(errors),
        "errors": errors,
    }
    return report, 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Wave 5 hypercomplex research scaffolding.")
    parser.add_argument(
        "--out-json",
        default=str(ROOT_DIR / "artifacts" / "research" / "hypercomplex_wave5_validation.v1.json"),
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
        "hypercomplex-wave5 validation passed: "
        f"{report['counts']['wave5_touchpoints']} wave5 touchpoint, "
        f"{report['counts']['compile_fixture_count']} compile fixture"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
