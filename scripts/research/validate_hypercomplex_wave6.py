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
WAVE5_VALIDATOR = ROOT_DIR / "scripts" / "research" / "validate_hypercomplex_wave5.py"

REQUIRED_WAVE5_TOUCHPOINTS = {
    "wave5_reassociate_metadata_compile_proof_harness",
}
REQUIRED_WAVE6_TOUCHPOINTS = {
    "wave6_forbidden_law_metadata_compile_proof_harness",
}
REQUIRED_WAVE5_AUDITS = {
    "ir.reassociate_strategy_compile_proof_audit",
}
REQUIRED_WAVE6_AUDITS = {
    "ir.forbidden_law_mask_compile_proof_audit",
}
REQUIRED_WAVE_SCAFFOLDING = {
    "wave5_reassociate_metadata_compile_proof_harness",
    "wave6_forbidden_law_metadata_compile_proof_harness",
}
REQUIRED_VALIDATION_CONTRACT_KEYS = {
    "required_inventory",
    "required_touchpoints",
    "required_semantics",
    "required_hazards",
    "forbidden_scaffolding_touchpoints",
    "notes",
}


def load_wave5_module():
    spec = importlib.util.spec_from_file_location("hypercomplex_wave5", WAVE5_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load wave5 validator from {WAVE5_VALIDATOR}")
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
    out_path = Path("/tmp") / (fixture_path.stem + ".wave6.elf")
    log_path = Path("/tmp") / (fixture_path.stem + ".wave6.log")
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

    wave5_module = load_wave5_module()
    wave5_report, wave5_code = wave5_module.validate()
    if wave5_code != 0:
        errors.append("wave5 hypercomplex validation failed")

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
            "schema": "sounio.research.hypercomplex.validation_report.v6",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "fail",
            "error_count": len(errors),
            "errors": errors,
            "wave5_status": wave5_report.get("status", "fail"),
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
    touchpoint_ids = set(touchpoint_by_id.keys())
    hazard_entries = hazards.get("hazards", [])
    hazard_by_id = {
        entry.get("id"): entry for entry in hazard_entries if isinstance(entry.get("id"), str)
    }
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
    semantic_topics = semantics.get("semantic_topics", [])
    semantic_ids = {entry.get("id") for entry in semantic_topics if isinstance(entry.get("id"), str)}

    prototype_safe_contract = semantics.get("prototype_safe_contract", [])
    check(
        any("validation-only" in item for item in prototype_safe_contract if isinstance(item, str)),
        "prototype_safe_contract must mention validation-only forbidden-law gaps",
        errors,
    )
    check(
        any("forbidden-law mask" in item for item in prototype_safe_contract if isinstance(item, str)),
        "prototype_safe_contract must mention forbidden-law mask carriage",
        errors,
    )

    missing_touchpoints = sorted((REQUIRED_WAVE5_TOUCHPOINTS | REQUIRED_WAVE6_TOUCHPOINTS) - touchpoint_ids)
    check(not missing_touchpoints, f"missing required touchpoint entries: {missing_touchpoints}", errors)

    missing_audits = sorted((REQUIRED_WAVE5_AUDITS | REQUIRED_WAVE6_AUDITS) - set(audit_by_id.keys()))
    check(not missing_audits, f"missing required compiler audit entries: {missing_audits}", errors)

    for entry_id in REQUIRED_WAVE5_TOUCHPOINTS | REQUIRED_WAVE6_TOUCHPOINTS:
        entry = touchpoint_by_id.get(entry_id, {})
        check(entry.get("classification") == "prototype-safe", f"touchpoint {entry_id} must be prototype-safe", errors)
        check(entry.get("landing_state") == "landed-validation", f"touchpoint {entry_id} must be landed-validation", errors)
        for ref in entry.get("references", []) if isinstance(entry.get("references"), list) else []:
            check(isinstance(ref, dict), f"touchpoint {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors, require_symbol_match=entry_id in REQUIRED_WAVE6_TOUCHPOINTS)

    for entry_id in REQUIRED_WAVE5_AUDITS | REQUIRED_WAVE6_AUDITS:
        entry = audit_by_id.get(entry_id, {})
        check(entry.get("classification") == "prototype-safe", f"compiler audit {entry_id} must be prototype-safe", errors)
        check(entry.get("landing_state") == "landed-validation", f"compiler audit {entry_id} must be landed-validation", errors)
        for ref in entry.get("references", []) if isinstance(entry.get("references"), list) else []:
            check(isinstance(ref, dict), f"compiler audit {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors, require_symbol_match=True)

    compile_fixture_paths: list[str] = []
    for entry_id in REQUIRED_WAVE_SCAFFOLDING:
        entry = scaffolding_by_id.get(entry_id, {})
        check(entry.get("classification") == "prototype-safe", f"prototype scaffolding {entry_id} must be prototype-safe", errors)
        check(entry.get("landing_state") == "landed-validation", f"prototype scaffolding {entry_id} must be landed-validation", errors)
        fixtures = entry.get("fixtures", []) if isinstance(entry, dict) else []
        check(isinstance(fixtures, list) and fixtures, f"prototype scaffolding {entry_id} must reference compile fixtures", errors)
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
            if isinstance(fixture_path, str) and fixture_path != "" and fixture_mode == "compile-pass":
                compile_fixture_paths.append(fixture_path)
        for ref in entry.get("references", []) if isinstance(entry.get("references"), list) else []:
            check(isinstance(ref, dict), f"prototype scaffolding {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors, require_symbol_match=True)

    gap_entry = expected_fail_by_id.get("generic_distribute_or_factor.contract_gap")
    check(isinstance(gap_entry, dict), "missing generic_distribute_or_factor.contract_gap entry", errors)
    validation_contract = gap_entry.get("validation_contract") if isinstance(gap_entry, dict) else None
    check(isinstance(validation_contract, dict), "generic_distribute_or_factor.contract_gap must include validation_contract", errors)
    if isinstance(validation_contract, dict):
        missing_keys = sorted(REQUIRED_VALIDATION_CONTRACT_KEYS - set(validation_contract.keys()))
        check(not missing_keys, f"generic_distribute_or_factor.contract_gap missing validation_contract keys: {missing_keys}", errors)
        notes = validation_contract.get("notes")
        check(isinstance(notes, str) and notes != "", "validation_contract notes must be non-empty", errors)
        check(
            isinstance(notes, str) and "forbidden-law masks remain advisory metadata only" in notes,
            "validation_contract notes must keep Wave 6 forbidden-law mask advisory note",
            errors,
        )
        required_semantics = validation_contract.get("required_semantics", [])
        for semantic_id in required_semantics if isinstance(required_semantics, list) else []:
            check(semantic_id in semantic_ids, f"validation_contract references unknown semantic id: {semantic_id}", errors)

    inventory_entry = inventory_by_id.get("ir.hyper_metadata_and_opcode_surface", {})
    check(
        "forbidden-law masks" in inventory_entry.get("summary", ""),
        "ir.hyper_metadata_and_opcode_surface summary must mention forbidden-law masks",
        errors,
    )

    boundary_touchpoint = touchpoint_by_id.get("internal_ir_annotations.boundary", {})
    check(
        "forbidden-law" in boundary_touchpoint.get("summary", ""),
        "internal_ir_annotations.boundary summary must mention forbidden-law metadata",
        errors,
    )

    assoc_hazard = hazard_by_id.get("associativity_sensitive_rewrites", {})
    assoc_refs = assoc_hazard.get("references", []) if isinstance(assoc_hazard, dict) else []
    check(
        any(
            isinstance(ref, dict)
            and ref.get("path") == "self-hosted/ir/algebra.sio"
            and ref.get("symbol") == "IR_HYPER_FORBID_FLATTEN_NONASSOC_MUL"
            for ref in assoc_refs
        ),
        "associativity_sensitive_rewrites must reference IR_HYPER_FORBID_FLATTEN_NONASSOC_MUL",
        errors,
    )

    dist_hazard = hazard_by_id.get("distributivity_sensitive_rewrites", {})
    dist_refs = dist_hazard.get("references", []) if isinstance(dist_hazard, dict) else []
    check(
        any(
            isinstance(ref, dict)
            and ref.get("path") == "self-hosted/ir/algebra.sio"
            and ref.get("symbol") == "IR_HYPER_FORBID_GENERIC_DISTRIBUTE_FACTOR"
            for ref in dist_refs
        ),
        "distributivity_sensitive_rewrites must reference IR_HYPER_FORBID_GENERIC_DISTRIBUTE_FACTOR",
        errors,
    )

    fixture_results: list[dict] = []
    unique_fixture_paths = sorted(set(compile_fixture_paths))
    if not errors and unique_fixture_paths:
        souc_bin = resolve_souc(errors)
        if souc_bin:
            for fixture_path in unique_fixture_paths:
                result = compile_fixture(souc_bin, fixture_path)
                fixture_results.append(result)
                if result["status"] != "pass":
                    errors.append(f"Wave 6 compile-proof fixture failed: {fixture_path}")

    readme_text = readme_path.read_text(encoding="utf-8")
    check("hypercomplex_wave6_gate.sh" in readme_text, "README must mention hypercomplex_wave6_gate.sh", errors)
    check("forbidden-law mask" in readme_text, "README must describe the Wave 6 forbidden-law mask seam", errors)

    report = {
        "schema": "sounio.research.hypercomplex.validation_report.v6",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass" if not errors else "fail",
        "wave5_status": wave5_report.get("status", "fail"),
        "files": {
            "prototype_scaffolding": "research/hypercomplex/prototype_scaffolding.v1.json",
            "expected_fail": "research/hypercomplex/expected_fail.v1.json",
            "compiler_audit": "research/hypercomplex/compiler_audit.v1.json",
            "touchpoints": "research/hypercomplex/touchpoints.v1.json",
            "semantics": "research/hypercomplex/semantics.v1.json",
            "hazards": "research/hypercomplex/hazards.v1.json",
            "readme": "research/hypercomplex/README.md",
        },
        "counts": {
            "wave5_touchpoints": len(REQUIRED_WAVE5_TOUCHPOINTS),
            "wave6_touchpoints": len(REQUIRED_WAVE6_TOUCHPOINTS),
            "wave5_audits": len(REQUIRED_WAVE5_AUDITS),
            "wave6_audits": len(REQUIRED_WAVE6_AUDITS),
            "compile_fixture_count": len(unique_fixture_paths),
        },
        "compile_fixtures": fixture_results,
        "error_count": len(errors),
        "errors": errors,
    }
    return report, 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Wave 6 hypercomplex research scaffolding.")
    parser.add_argument(
        "--out-json",
        default=str(ROOT_DIR / "artifacts" / "research" / "hypercomplex_wave6_validation.v1.json"),
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
        "hypercomplex-wave6 validation passed: "
        f"{report['counts']['wave6_touchpoints']} wave6 touchpoint, "
        f"{report['counts']['wave6_audits']} wave6 audit, "
        f"{report['counts']['compile_fixture_count']} compile fixtures"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
