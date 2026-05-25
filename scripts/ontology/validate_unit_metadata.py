#!/usr/bin/env python3
"""Validate ontology-linked internal dimension-label metadata.

This is a data-quality fixture gate, not a PL reasoning path, clinical
authority, conversion authority, or terminology-conformance certificate. The
Sounio checker and UnitRegistry remain the authority for internal unit
semantics; this script only proves that metadata rows reference known local
fixture terms, declared unit atoms, and registered Sounio unit identifiers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


EXPECTED_SCHEMA = "sounio.ontology.internal-dimension-labels.v1"
EXPECTED_STATUS = "validation-data"
ALLOWED_FAMILIES = {
    "amount_concentration",
    "enzyme_activity_concentration",
    "length_per_time",
    "mass_concentration",
    "ratio",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_terms(path: Path) -> dict[str, str]:
    doc = load_json(path)
    out = {}
    for term in doc.get("terms", []):
        curie = term.get("curie")
        label = term.get("label")
        if curie and label:
            out[curie] = label
    return out


def canonical_unit_atoms(unit: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|percent", unit)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Exit codes: 0 success; 1 validation failure; 2 non-authoritative "
            "metadata refused because --accept-no-clinical-safety was missing."
        ),
    )
    parser.add_argument(
        "--metadata",
        default=(
            "stdlib/data/data/ontology/unit_metadata/"
            "internal_unit_labels_no_clinical_dimension_safety.json"
        ),
    )
    parser.add_argument(
        "--loinc-source",
        default="stdlib/data/data/ontology/source/loinc_slice.json",
    )
    parser.add_argument(
        "--chebi-source",
        default="stdlib/data/data/ontology/source/chebi_slice.json",
    )
    parser.add_argument(
        "--lean-single",
        default="self-hosted/compiler/lean_single.sio",
    )
    parser.add_argument(
        "--accept-no-clinical-safety",
        action="store_true",
        help=(
            "Acknowledge checker_authority=false and that this metadata asserts "
            "no clinical safety, conversion safety, or standards conformance. "
            "Without this flag, non-authoritative metadata is a hard failure."
        ),
    )
    parser.add_argument(
        "--authority-warning-log",
        default="/tmp/sounio-internal-label-authority-warning.json",
        help="Path for a structured warning record when checker_authority=false.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    doc = load_json(metadata_path)
    if doc.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"{metadata_path} has wrong schema: {doc.get('schema')!r}")
    if doc.get("status") != EXPECTED_STATUS:
        raise ValueError(f"{metadata_path} must be status={EXPECTED_STATUS!r}")
    if doc.get("checker_authority") is not False:
        raise ValueError(f"{metadata_path} must set checker_authority=false")
    warning = (
        "WARNING checker_authority=false: metadata is internal fixture "
        "well-formedness only; no clinical validation, conversion authority, "
        "or terminology conformance is implied."
    )
    print(warning, file=sys.stderr)
    warning_record = {
        "metadata": str(metadata_path),
        "checker_authority": False,
        "machine_marker": "@sounio/safety:no-clinical-dimension",
        "message": warning,
        "required_acknowledgements": [
            "--accept-no-clinical-safety",
        ],
    }
    Path(args.authority_warning_log).write_text(
        json.dumps(warning_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.accept_no_clinical_safety:
        print(
            "ERROR refusing non-authoritative metadata without "
            "--accept-no-clinical-safety",
            file=sys.stderr,
        )
        print(
            f"Structured warning written to {args.authority_warning_log}",
            file=sys.stderr,
        )
        return 2
    safety_notice = doc.get("clinical_safety_notice", "")
    if "no clinical validation has been performed" not in safety_notice:
        raise ValueError(
            f"{metadata_path} must explicitly state that checker_authority=false "
            "means no clinical validation has been performed"
        )
    safety_contract = doc.get("safety_contract")
    if not isinstance(safety_contract, dict):
        raise ValueError(f"{metadata_path} must declare safety_contract")
    if safety_contract.get("machine_marker") != "@sounio/safety:no-clinical-dimension":
        raise ValueError(f"{metadata_path} must carry @sounio/safety:no-clinical-dimension")
    if safety_contract.get("scope") != "fixture_well_formedness":
        raise ValueError(f"{metadata_path} safety_contract.scope must be fixture_well_formedness")
    if safety_contract.get("enforcement") != "flag_gated_fixture_validation":
        raise ValueError(
            f"{metadata_path} safety_contract.enforcement must be "
            "flag_gated_fixture_validation"
        )
    if safety_contract.get("acknowledgement_flag") != "--accept-no-clinical-safety":
        raise ValueError(
            f"{metadata_path} safety_contract.acknowledgement_flag must be "
            "--accept-no-clinical-safety"
        )
    excluded = set(safety_contract.get("excluded", []))
    required_exclusions = {
        "clinical_data_quality",
        "identifier_correctness",
        "conversion_correctness",
        "standards_conformance",
        "dosing_safety",
        "regulatory_exchange",
    }
    if not required_exclusions.issubset(excluded):
        raise ValueError(
            f"{metadata_path} safety_contract.excluded missing "
            f"{sorted(required_exclusions - excluded)}"
        )

    loinc_terms = load_terms(Path(args.loinc_source))
    chebi_terms = load_terms(Path(args.chebi_source))

    unit_atoms = doc.get("unit_atoms", [])
    if not unit_atoms:
        raise ValueError(f"{metadata_path} has no unit_atoms")
    known_atoms = set()
    builtin_atoms = set()
    for atom in unit_atoms:
        symbol = atom.get("symbol")
        dimension = atom.get("dimension")
        if not symbol or not dimension:
            raise ValueError(f"{metadata_path} has incomplete unit atom {atom!r}")
        if symbol in known_atoms:
            raise ValueError(f"{metadata_path} has duplicate unit atom {symbol!r}")
        known_atoms.add(symbol)
        if atom.get("sounio_builtin") is True:
            builtin_atoms.add(symbol)

    required_builtin_atoms = {"mg", "g", "mol", "mm", "h"}
    missing_builtin = required_builtin_atoms - builtin_atoms
    if missing_builtin:
        raise ValueError(f"{metadata_path} missing builtin unit atoms: {sorted(missing_builtin)}")

    checker_unit_hashes = {
        "mg_dL": "122913530894",
        "mmol_L": "8514995863",
        "U_L": "7844023011",
        "mm_h": "125392651336",
    }
    lean_single_source = Path(args.lean_single).read_text(encoding="utf-8")

    entries = doc.get("entries", [])
    if len(entries) < 8:
        raise ValueError(f"{metadata_path} should contain at least 8 internal label entries")

    seen = set()
    chebi_linked = 0
    for entry in entries:
        loinc_curie = entry.get("loinc_curie")
        loinc_label = entry.get("loinc_label")
        chebi_curie = entry.get("chebi_curie")
        chebi_label = entry.get("chebi_label")
        family = entry.get("unit_family")
        canonical_unit = entry.get("canonical_unit")
        sounio_unit_name = entry.get("sounio_unit_name")

        if loinc_curie not in loinc_terms:
            raise ValueError(f"{metadata_path} references unknown LOINC term {loinc_curie!r}")
        if loinc_terms[loinc_curie] != loinc_label:
            raise ValueError(
                f"{metadata_path} label mismatch for {loinc_curie}: "
                f"{loinc_label!r} != {loinc_terms[loinc_curie]!r}"
            )
        if family not in ALLOWED_FAMILIES:
            raise ValueError(f"{metadata_path} has unsupported unit_family {family!r}")
        if not canonical_unit or not sounio_unit_name:
            raise ValueError(f"{metadata_path} has incomplete unit entry {entry!r}")
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", sounio_unit_name):
            raise ValueError(f"{metadata_path} has invalid sounio_unit_name {sounio_unit_name!r}")
        if sounio_unit_name != "percent" and sounio_unit_name not in checker_unit_hashes:
            raise ValueError(
                f"{metadata_path} sounio_unit_name lacks checker registration evidence: "
                f"{sounio_unit_name!r}"
            )
        if sounio_unit_name in checker_unit_hashes:
            marker = f"unit_register_h({checker_unit_hashes[sounio_unit_name]},"
            if marker not in lean_single_source:
                raise ValueError(
                    f"{metadata_path} sounio_unit_name {sounio_unit_name!r} is not "
                    f"registered in {args.lean_single}"
                )

        for atom in canonical_unit_atoms(canonical_unit):
            if atom not in known_atoms:
                raise ValueError(
                    f"{metadata_path} entry {loinc_curie} uses undeclared unit atom {atom!r}"
                )

        if chebi_curie is not None:
            chebi_linked += 1
            if chebi_curie not in chebi_terms:
                raise ValueError(f"{metadata_path} references unknown ChEBI term {chebi_curie!r}")
            if chebi_terms[chebi_curie] != chebi_label:
                raise ValueError(
                    f"{metadata_path} label mismatch for {chebi_curie}: "
                    f"{chebi_label!r} != {chebi_terms[chebi_curie]!r}"
                )

        key = (loinc_curie, canonical_unit)
        if key in seen:
            raise ValueError(f"{metadata_path} has duplicate LOINC/unit entry {key!r}")
        seen.add(key)

    if chebi_linked < 4:
        raise ValueError(f"{metadata_path} should link at least 4 entries to ChEBI terms")

    print(
        "ontology internal dimension-label metadata: OK "
        f"entries={len(entries)} unit_atoms={len(unit_atoms)} chebi_links={chebi_linked}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
