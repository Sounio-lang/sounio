#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "research" / "hypercomplex"
WAVE1_VALIDATOR = ROOT_DIR / "scripts" / "research" / "validate_hypercomplex_wave1.py"

REQUIRED_SEMANTIC_TOPICS = {
    "parenthesization_sensitivity",
    "associativity_assumptions",
    "distributivity_assumptions",
    "zero_divisor_sensitivity",
    "inverse_division_domain_restrictions",
    "norm_conjugation_assumptions",
    "expression_reorderability",
    "optimizer_law_safety",
}

REQUIRED_CANDIDATE_AREAS = {
    "type-metadata-tagging",
    "internal-ir-annotations",
    "parenthesization-sensitive-fixtures",
    "expected-fail-law-fixtures",
    "non-public-validation-scaffolding",
}

REQUIRED_LANDING_STATES = {"mapping-only", "landed-validation", "deferred"}
REQUIRED_EXPERIMENT_CLASSES = {
    "safe_compile_proof_prototype",
    "unsafe_without_formal_semantics",
    "requires_new_runtime_abi",
}


def load_wave1_module():
    spec = importlib.util.spec_from_file_location("hypercomplex_wave1", WAVE1_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load wave1 validator from {WAVE1_VALIDATOR}")
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


def validate() -> tuple[dict, int]:
    errors: list[str] = []

    wave1_module = load_wave1_module()
    wave1_report, wave1_code = wave1_module.validate()
    if wave1_code != 0:
        errors.append("wave1 hypercomplex validation failed")

    taxonomy = load_json(DATA_DIR / "taxonomy.v1.json", errors)
    inventory = load_json(DATA_DIR / "inventory.v1.json", errors)
    hazards = load_json(DATA_DIR / "hazards.v1.json", errors)
    semantics = load_json(DATA_DIR / "semantics.v1.json", errors)
    touchpoints = load_json(DATA_DIR / "touchpoints.v1.json", errors)
    roadmap = load_json(DATA_DIR / "roadmap.v1.json", errors)
    readme_path = DATA_DIR / "README.md"
    check(readme_path.exists(), f"missing required file: {readme_path.relative_to(ROOT_DIR)}", errors)

    if errors:
        report = {
            "schema": "sounio.research.hypercomplex.validation_report.v2",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "fail",
            "error_count": len(errors),
            "errors": errors,
            "wave1_status": wave1_report.get("status", "fail"),
        }
        return report, 1

    classes = taxonomy.get("classes", {})
    inventory_entries = inventory.get("entries", [])
    hazard_entries = hazards.get("hazards", [])
    semantic_topics = semantics.get("semantic_topics", [])
    forbidden_laws = semantics.get("forbidden_laws", [])
    prototype_safe_contract = semantics.get("prototype_safe_contract", [])
    touchpoint_entries = touchpoints.get("entries", [])
    candidate_areas = touchpoints.get("candidate_areas", [])
    landing_states = touchpoints.get("landing_states", [])
    roadmap_experiments = roadmap.get("experiments", [])
    experiment_classes = roadmap.get("experiment_classes", {})

    check(semantics.get("schema") == "sounio.research.hypercomplex.semantics.v1", "unexpected semantics schema", errors)
    check(touchpoints.get("schema") == "sounio.research.hypercomplex.touchpoints.v1", "unexpected touchpoints schema", errors)
    check(roadmap.get("schema") == "sounio.research.hypercomplex.roadmap.v1", "unexpected roadmap schema", errors)

    check(isinstance(classes, dict) and classes, "taxonomy classes must be available from wave1", errors)
    allowed_classes = set(classes.keys()) if isinstance(classes, dict) else set()

    inventory_ids = {entry.get("id") for entry in inventory_entries if isinstance(entry.get("id"), str)}
    hazard_ids = {hazard.get("id") for hazard in hazard_entries if isinstance(hazard.get("id"), str)}

    check(isinstance(prototype_safe_contract, list) and prototype_safe_contract, "prototype_safe_contract must be a non-empty list", errors)
    check(set(candidate_areas) == REQUIRED_CANDIDATE_AREAS, f"candidate_areas mismatch: expected {sorted(REQUIRED_CANDIDATE_AREAS)}", errors)
    check(set(landing_states) == REQUIRED_LANDING_STATES, f"landing_states mismatch: expected {sorted(REQUIRED_LANDING_STATES)}", errors)
    check(set(experiment_classes.keys()) == REQUIRED_EXPERIMENT_CLASSES, f"experiment_classes mismatch: expected {sorted(REQUIRED_EXPERIMENT_CLASSES)}", errors)

    seen_semantic_ids: set[str] = set()
    semantic_counts = {cls: 0 for cls in allowed_classes}
    for entry in semantic_topics:
        topic_id = entry.get("id")
        classification = entry.get("classification")
        hazards_list = entry.get("hazards", [])
        inventory_list = entry.get("inventory", [])
        refs = entry.get("references", [])

        check(isinstance(topic_id, str) and topic_id != "", "semantic topic missing id", errors)
        if not isinstance(topic_id, str) or topic_id == "":
            continue
        check(topic_id not in seen_semantic_ids, f"duplicate semantic topic id: {topic_id}", errors)
        seen_semantic_ids.add(topic_id)

        check(classification in allowed_classes, f"semantic topic {topic_id} has invalid classification: {classification}", errors)
        if classification in semantic_counts:
            semantic_counts[classification] += 1
        check(isinstance(entry.get("summary"), str) and entry["summary"] != "", f"semantic topic {topic_id} missing summary", errors)
        check(isinstance(entry.get("operational_contract"), str) and entry["operational_contract"] != "", f"semantic topic {topic_id} missing operational_contract", errors)
        check(isinstance(hazards_list, list) and hazards_list, f"semantic topic {topic_id} must reference hazards", errors)
        check(isinstance(inventory_list, list) and inventory_list, f"semantic topic {topic_id} must reference inventory ids", errors)
        check(isinstance(refs, list) and refs, f"semantic topic {topic_id} must reference source files", errors)

        for hazard_id in hazards_list if isinstance(hazards_list, list) else []:
            check(hazard_id in hazard_ids, f"semantic topic {topic_id} references unknown hazard id: {hazard_id}", errors)
        for inventory_id in inventory_list if isinstance(inventory_list, list) else []:
            check(inventory_id in inventory_ids, f"semantic topic {topic_id} references unknown inventory id: {inventory_id}", errors)
        for ref in refs if isinstance(refs, list) else []:
            check(isinstance(ref, dict), f"semantic topic {topic_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors)

    missing_topics = sorted(REQUIRED_SEMANTIC_TOPICS - seen_semantic_ids)
    check(not missing_topics, f"missing required semantic topics: {missing_topics}", errors)

    seen_law_ids: set[str] = set()
    for law in forbidden_laws:
        law_id = law.get("id")
        classification = law.get("classification")
        topics = law.get("topics", [])
        refs = law.get("references", [])

        check(isinstance(law_id, str) and law_id != "", "forbidden law missing id", errors)
        if not isinstance(law_id, str) or law_id == "":
            continue
        check(law_id not in seen_law_ids, f"duplicate forbidden law id: {law_id}", errors)
        seen_law_ids.add(law_id)

        check(classification in allowed_classes, f"forbidden law {law_id} has invalid classification: {classification}", errors)
        check(isinstance(law.get("summary"), str) and law["summary"] != "", f"forbidden law {law_id} missing summary", errors)
        check(isinstance(topics, list) and topics, f"forbidden law {law_id} must reference semantic topics", errors)
        check(isinstance(refs, list) and refs, f"forbidden law {law_id} must reference source files", errors)

        for topic_id in topics if isinstance(topics, list) else []:
            check(topic_id in seen_semantic_ids, f"forbidden law {law_id} references unknown semantic topic: {topic_id}", errors)
        for ref in refs if isinstance(refs, list) else []:
            check(isinstance(ref, dict), f"forbidden law {law_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors)

    seen_touchpoint_ids: set[str] = set()
    seen_candidate_areas: set[str] = set()
    touchpoint_counts = {cls: 0 for cls in allowed_classes}
    for entry in touchpoint_entries:
        entry_id = entry.get("id")
        candidate_area = entry.get("candidate_area")
        classification = entry.get("classification")
        landing_state = entry.get("landing_state")
        inventory_list = entry.get("inventory", [])
        semantics_list = entry.get("semantics", [])
        refs = entry.get("references", [])

        check(isinstance(entry_id, str) and entry_id != "", "touchpoint entry missing id", errors)
        if not isinstance(entry_id, str) or entry_id == "":
            continue
        check(entry_id not in seen_touchpoint_ids, f"duplicate touchpoint id: {entry_id}", errors)
        seen_touchpoint_ids.add(entry_id)

        check(candidate_area in REQUIRED_CANDIDATE_AREAS, f"touchpoint {entry_id} has invalid candidate_area: {candidate_area}", errors)
        if candidate_area in REQUIRED_CANDIDATE_AREAS:
            seen_candidate_areas.add(candidate_area)
        check(classification in allowed_classes, f"touchpoint {entry_id} has invalid classification: {classification}", errors)
        if classification in touchpoint_counts:
            touchpoint_counts[classification] += 1
        check(landing_state in REQUIRED_LANDING_STATES, f"touchpoint {entry_id} has invalid landing_state: {landing_state}", errors)
        check(isinstance(entry.get("summary"), str) and entry["summary"] != "", f"touchpoint {entry_id} missing summary", errors)
        check(isinstance(inventory_list, list) and inventory_list, f"touchpoint {entry_id} must reference inventory ids", errors)
        check(isinstance(semantics_list, list) and semantics_list, f"touchpoint {entry_id} must reference semantics ids", errors)
        check(isinstance(refs, list) and refs, f"touchpoint {entry_id} must reference source files", errors)

        for inventory_id in inventory_list if isinstance(inventory_list, list) else []:
            check(inventory_id in inventory_ids, f"touchpoint {entry_id} references unknown inventory id: {inventory_id}", errors)
        for semantic_id in semantics_list if isinstance(semantics_list, list) else []:
            check(semantic_id in seen_semantic_ids, f"touchpoint {entry_id} references unknown semantic topic: {semantic_id}", errors)
        for ref in refs if isinstance(refs, list) else []:
            check(isinstance(ref, dict), f"touchpoint {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors)

    missing_candidate_areas = sorted(REQUIRED_CANDIDATE_AREAS - seen_candidate_areas)
    check(not missing_candidate_areas, f"missing touchpoint coverage for candidate areas: {missing_candidate_areas}", errors)

    seen_experiment_ids: set[str] = set()
    seen_experiment_classes: set[str] = set()
    for experiment in roadmap_experiments:
        experiment_id = experiment.get("id")
        experiment_class = experiment.get("experiment_class")
        linked_touchpoints = experiment.get("touchpoints", [])

        check(isinstance(experiment_id, str) and experiment_id != "", "roadmap experiment missing id", errors)
        if not isinstance(experiment_id, str) or experiment_id == "":
            continue
        check(experiment_id not in seen_experiment_ids, f"duplicate roadmap experiment id: {experiment_id}", errors)
        seen_experiment_ids.add(experiment_id)

        check(experiment_class in REQUIRED_EXPERIMENT_CLASSES, f"roadmap experiment {experiment_id} has invalid class: {experiment_class}", errors)
        if experiment_class in REQUIRED_EXPERIMENT_CLASSES:
            seen_experiment_classes.add(experiment_class)
        check(isinstance(experiment.get("summary"), str) and experiment["summary"] != "", f"roadmap experiment {experiment_id} missing summary", errors)
        check(isinstance(experiment.get("why_now"), str) and experiment["why_now"] != "", f"roadmap experiment {experiment_id} missing why_now", errors)
        check(isinstance(linked_touchpoints, list) and linked_touchpoints, f"roadmap experiment {experiment_id} must reference touchpoints", errors)

        for touchpoint_id in linked_touchpoints if isinstance(linked_touchpoints, list) else []:
            check(touchpoint_id in seen_touchpoint_ids, f"roadmap experiment {experiment_id} references unknown touchpoint id: {touchpoint_id}", errors)

    missing_experiment_classes = sorted(REQUIRED_EXPERIMENT_CLASSES - seen_experiment_classes)
    check(not missing_experiment_classes, f"missing roadmap coverage for experiment classes: {missing_experiment_classes}", errors)

    report = {
        "schema": "sounio.research.hypercomplex.validation_report.v2",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass" if not errors else "fail",
        "wave1_status": wave1_report.get("status", "fail"),
        "files": {
            "taxonomy": "research/hypercomplex/taxonomy.v1.json",
            "inventory": "research/hypercomplex/inventory.v1.json",
            "hazards": "research/hypercomplex/hazards.v1.json",
            "semantics": "research/hypercomplex/semantics.v1.json",
            "touchpoints": "research/hypercomplex/touchpoints.v1.json",
            "roadmap": "research/hypercomplex/roadmap.v1.json",
            "readme": "research/hypercomplex/README.md"
        },
        "counts": {
            "semantic_topics": len(semantic_topics),
            "forbidden_laws": len(forbidden_laws),
            "touchpoint_entries": len(touchpoint_entries),
            "roadmap_experiments": len(roadmap_experiments),
            "semantic_classifications": semantic_counts,
            "touchpoint_classifications": touchpoint_counts
        },
        "error_count": len(errors),
        "errors": errors
    }
    return report, 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Wave 2 hypercomplex research manifests.")
    parser.add_argument(
        "--out-json",
        default=str(ROOT_DIR / "artifacts" / "research" / "hypercomplex_wave2_validation.v1.json"),
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
        "hypercomplex-wave2 validation passed: "
        f"{report['counts']['semantic_topics']} semantic topics, "
        f"{report['counts']['touchpoint_entries']} touchpoints, "
        f"{report['counts']['roadmap_experiments']} roadmap experiments"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
