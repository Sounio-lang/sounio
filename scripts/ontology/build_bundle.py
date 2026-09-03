#!/usr/bin/env python3
"""Build reproducible .dontology bundles from source JSON fixtures.

The binary layout is intentionally simple and stable for phase 1:

- 4 bytes magic: DONT
- u32 version: 1
- u32 payload length
- payload: UTF-8 encoded canonical JSON
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


MAGIC = b"DONT"
VERSION = 1


def canonical_json(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_bundle_payload(source_doc: dict[str, Any]) -> dict[str, Any]:
    terms = source_doc.get("terms", [])
    disjoints = source_doc.get("disjoints", [])
    return {
        "ontology": source_doc["ontology"],
        "version": source_doc.get("version", "unknown"),
        "term_count": len(terms),
        "disjoint_count": len(disjoints),
        "terms": terms,
        "disjoints": disjoints,
    }


def write_bundle(output_path: Path, payload: dict[str, Any]) -> None:
    body = canonical_json(payload)
    header = MAGIC + struct.pack("<II", VERSION, len(body))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + body)


def build_all(source_dir: Path, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    for source_path in sorted(source_dir.glob("*_slice.json")):
        source_doc = json.loads(source_path.read_text())
        ontology_name = source_doc["ontology"].lower()
        output_path = out_dir / f"{ontology_name}.dontology"
        write_bundle(output_path, build_bundle_payload(source_doc))
        written.append(output_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="data/ontology/source",
        help="Directory containing source *_slice.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/ontology/bundles",
        help="Directory where .dontology bundles will be written.",
    )
    args = parser.parse_args()

    written = build_all(Path(args.source_dir), Path(args.output_dir))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
