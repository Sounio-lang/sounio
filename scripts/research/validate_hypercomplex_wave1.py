#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "research" / "hypercomplex"


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


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR))


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

    taxonomy_path = DATA_DIR / "taxonomy.v1.json"
    inventory_path = DATA_DIR / "inventory.v1.json"
    hazards_path = DATA_DIR / "hazards.v1.json"
    readme_path = DATA_DIR / "README.md"

    check(readme_path.exists(), f"missing required file: {rel(readme_path)}", errors)

    taxonomy = load_json(taxonomy_path, errors)
    inventory = load_json(inventory_path, errors)
    hazards = load_json(hazards_path, errors)

    if errors:
      report = {
          "schema": "sounio.research.hypercomplex.validation_report.v1",
          "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
          "status": "fail",
          "error_count": len(errors),
          "errors": errors
      }
      return report, 1

    check(
        taxonomy.get("schema") == "sounio.research.hypercomplex.taxonomy.v1",
        "unexpected taxonomy schema",
        errors,
    )
    check(
        inventory.get("schema") == "sounio.research.hypercomplex.inventory.v1",
        "unexpected inventory schema",
        errors,
    )
    check(
        hazards.get("schema") == "sounio.research.hypercomplex.hazards.v1",
        "unexpected hazards schema",
        errors,
    )

    classes = taxonomy.get("classes", {})
    required_classes = {"research-only", "prototype-safe", "production-deferred"}
    check(isinstance(classes, dict), "taxonomy classes must be an object", errors)
    if isinstance(classes, dict):
        missing_classes = sorted(required_classes - set(classes))
        check(not missing_classes, f"taxonomy missing required classes: {missing_classes}", errors)

    categories = taxonomy.get("touchpoint_categories", [])
    states = taxonomy.get("surface_states", [])
    check(isinstance(categories, list) and categories, "taxonomy touchpoint_categories must be a non-empty list", errors)
    check(isinstance(states, list) and states, "taxonomy surface_states must be a non-empty list", errors)
    category_set = set(categories) if isinstance(categories, list) else set()
    state_set = set(states) if isinstance(states, list) else set()

    inventory_entries = inventory.get("entries", [])
    hazard_entries = hazards.get("hazards", [])
    check(isinstance(inventory_entries, list) and inventory_entries, "inventory entries must be a non-empty list", errors)
    check(isinstance(hazard_entries, list) and hazard_entries, "hazards must be a non-empty list", errors)

    inventory_ids: set[str] = set()
    inventory_state_counts = {state: 0 for state in state_set}
    inventory_class_counts = {cls: 0 for cls in required_classes}
    referenced_hazard_ids: set[str] = set()

    for entry in inventory_entries:
        entry_id = entry.get("id")
        check(isinstance(entry_id, str) and entry_id != "", "inventory entry missing id", errors)
        if not isinstance(entry_id, str) or entry_id == "":
            continue
        check(entry_id not in inventory_ids, f"duplicate inventory id: {entry_id}", errors)
        inventory_ids.add(entry_id)

        category = entry.get("category")
        classification = entry.get("classification")
        state = entry.get("surface_state")
        hazards_list = entry.get("hazards", [])
        refs = entry.get("references", [])

        check(category in category_set, f"inventory entry {entry_id} has unknown category: {category}", errors)
        check(classification in required_classes, f"inventory entry {entry_id} has invalid classification: {classification}", errors)
        check(state in state_set, f"inventory entry {entry_id} has invalid surface_state: {state}", errors)
        check(isinstance(hazards_list, list) and hazards_list, f"inventory entry {entry_id} must list at least one hazard", errors)
        check(isinstance(refs, list) and refs, f"inventory entry {entry_id} must list references", errors)

        if state in inventory_state_counts:
            inventory_state_counts[state] += 1
        if classification in inventory_class_counts:
            inventory_class_counts[classification] += 1

        for hazard_id in hazards_list if isinstance(hazards_list, list) else []:
            check(isinstance(hazard_id, str) and hazard_id != "", f"inventory entry {entry_id} has invalid hazard id", errors)
            if isinstance(hazard_id, str) and hazard_id:
                referenced_hazard_ids.add(hazard_id)

        for ref in refs if isinstance(refs, list) else []:
            check(isinstance(ref, dict), f"inventory entry {entry_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors)

    hazard_ids: set[str] = set()
    touched_inventory_ids: set[str] = set()
    for hazard in hazard_entries:
        hazard_id = hazard.get("id")
        classification = hazard.get("classification")
        touchpoints = hazard.get("touchpoints", [])
        refs = hazard.get("references", [])

        check(isinstance(hazard_id, str) and hazard_id != "", "hazard missing id", errors)
        if not isinstance(hazard_id, str) or hazard_id == "":
            continue
        check(hazard_id not in hazard_ids, f"duplicate hazard id: {hazard_id}", errors)
        hazard_ids.add(hazard_id)

        check(classification in required_classes, f"hazard {hazard_id} has invalid classification: {classification}", errors)
        check(isinstance(touchpoints, list) and touchpoints, f"hazard {hazard_id} must reference at least one inventory id", errors)
        check(isinstance(refs, list) and refs, f"hazard {hazard_id} must list references", errors)

        for touchpoint_id in touchpoints if isinstance(touchpoints, list) else []:
            check(touchpoint_id in inventory_ids, f"hazard {hazard_id} references unknown inventory id: {touchpoint_id}", errors)
            if touchpoint_id in inventory_ids:
                touched_inventory_ids.add(touchpoint_id)

        for ref in refs if isinstance(refs, list) else []:
            check(isinstance(ref, dict), f"hazard {hazard_id} has malformed reference", errors)
            if isinstance(ref, dict):
                validate_reference(ref, errors)

    missing_hazard_defs = sorted(referenced_hazard_ids - hazard_ids)
    check(not missing_hazard_defs, f"inventory references undefined hazards: {missing_hazard_defs}", errors)

    orphan_inventory = sorted(inventory_ids - touched_inventory_ids)
    check(not orphan_inventory, f"inventory entries without hazard coverage: {orphan_inventory}", errors)

    summary = inventory.get("summary", {})
    check(isinstance(summary, dict), "inventory summary must be an object", errors)
    if isinstance(summary, dict):
        expected_counts = {
            "active_entries": inventory_state_counts.get("active", 0),
            "historical_entries": inventory_state_counts.get("historical", 0),
            "contradictory_entries": inventory_state_counts.get("contradictory", 0),
            "absent_entries": inventory_state_counts.get("absent", 0),
        }
        for key, expected in expected_counts.items():
            check(summary.get(key) == expected, f"inventory summary mismatch for {key}: expected {expected}, found {summary.get(key)}", errors)

    report = {
        "schema": "sounio.research.hypercomplex.validation_report.v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass" if not errors else "fail",
        "files": {
            "taxonomy": rel(taxonomy_path),
            "inventory": rel(inventory_path),
            "hazards": rel(hazards_path),
            "readme": rel(readme_path),
        },
        "counts": {
            "inventory_entries": len(inventory_entries),
            "hazards": len(hazard_entries),
            "classifications": inventory_class_counts,
            "surface_states": inventory_state_counts,
        },
        "error_count": len(errors),
        "errors": errors,
    }
    return report, 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Wave 1 hypercomplex research manifests.")
    parser.add_argument(
        "--out-json",
        default=str(ROOT_DIR / "artifacts" / "research" / "hypercomplex_wave1_validation.v1.json"),
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
        "hypercomplex-wave1 validation passed: "
        f"{report['counts']['inventory_entries']} inventory entries, "
        f"{report['counts']['hazards']} hazards"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
