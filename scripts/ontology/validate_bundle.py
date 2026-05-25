#!/usr/bin/env python3
"""Validate ontology bundles and mapping shards.

Checks:
- correct .dontology magic/version
- payload is valid JSON
- required ontologies exist
- mapping shards contain confidence-bearing mappings
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


MAGIC = b"DONT"
VERSION = 1
REQUIRED = {"snomed", "loinc", "hpo", "go", "chebi"}


def read_bundle(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < 12:
        raise ValueError(f"{path} is too short to be a bundle")
    magic = raw[:4]
    if magic != MAGIC:
        raise ValueError(f"{path} has wrong magic {magic!r}")
    version, size = struct.unpack("<II", raw[4:12])
    if version != VERSION:
        raise ValueError(f"{path} has unsupported version {version}")
    payload = raw[12 : 12 + size]
    if len(payload) != size:
        raise ValueError(f"{path} payload length mismatch")
    return json.loads(payload.decode("utf-8"))


def validate_bundle_payload(path: Path) -> str:
    payload = read_bundle(path)
    ontology = payload.get("ontology", "").lower()
    if not ontology:
        raise ValueError(f"{path} missing ontology field")
    if payload.get("term_count", 0) <= 0:
        raise ValueError(f"{path} has no terms")
    terms = payload.get("terms", [])
    term_curies = {term.get("curie") for term in terms}
    if payload.get("term_count") != len(terms):
        raise ValueError(f"{path} term_count does not match terms")
    disjoints = payload.get("disjoints", [])
    if payload.get("disjoint_count", len(disjoints)) != len(disjoints):
        raise ValueError(f"{path} disjoint_count does not match disjoints")
    for pair in disjoints:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{path} has invalid disjoint pair {pair!r}")
        left, right = pair
        if left == right:
            raise ValueError(f"{path} has reflexive disjoint pair {pair!r}")
        if left not in term_curies or right not in term_curies:
            raise ValueError(f"{path} has disjoint pair with unknown term {pair!r}")
    return ontology


def validate_bundle_dir(bundle_dir: Path) -> None:
    found = set()
    for path in sorted(bundle_dir.glob("*.dontology")):
        found.add(validate_bundle_payload(path))

    missing = REQUIRED - found
    if missing:
        raise ValueError(f"missing required ontology bundles: {sorted(missing)}")


def validate_mapping_dir(mapping_dir: Path) -> None:
    mapping_files = list(mapping_dir.glob("*.sssom.json"))
    if not mapping_files:
        raise ValueError("no mapping shards found")
    for path in mapping_files:
        doc = json.loads(path.read_text())
        mappings = doc.get("mappings", [])
        if not mappings:
            raise ValueError(f"{path} has no mappings")
        for mapping in mappings:
            if "confidence" not in mapping:
                raise ValueError(f"{path} has mapping without confidence")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default="data/ontology/bundles")
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Validate one explicit .dontology bundle. May be repeated.",
    )
    parser.add_argument("--mapping-dir", default="data/ontology/bundles/mappings")
    args = parser.parse_args()

    if args.bundle:
        for bundle in args.bundle:
            validate_bundle_payload(Path(bundle))
    else:
        validate_bundle_dir(Path(args.bundle_dir))
    validate_mapping_dir(Path(args.mapping_dir))
    print("ontology bundles: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
