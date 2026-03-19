#!/usr/bin/env python3
"""Build SSSOM-style mapping shards for ontology interoperability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2) + "\n"


def build_all(source_dir: Path, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    for source_path in sorted(source_dir.glob("*.json")):
        doc = json.loads(source_path.read_text())
        out_path = out_dir / f"{source_path.stem}.sssom.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(canonical_json(doc))
        written.append(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="data/ontology/source/mappings",
        help="Directory containing source JSON mapping sets.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/ontology/bundles/mappings",
        help="Directory where mapping shards will be written.",
    )
    args = parser.parse_args()

    written = build_all(Path(args.source_dir), Path(args.output_dir))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
