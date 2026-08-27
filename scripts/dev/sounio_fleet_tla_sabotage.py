#!/usr/bin/env python3
"""Generate the executable sabotage catalog from the TLA+ fleet model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ANNOTATION = re.compile(
    r"^\\\* @sabotage id=(?P<id>[a-z0-9-]+) "
    r"invariant=(?P<invariant>[A-Za-z0-9_]+) "
    r"control=(?P<control>[a-z0-9_]+)$"
)


class CatalogError(RuntimeError):
    pass


def configured_invariants(path: Path) -> set[str]:
    names: set[str] = set()
    in_invariants = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "INVARIANTS":
            in_invariants = True
            continue
        if in_invariants and line and line.split()[0].isupper():
            break
        if in_invariants and line:
            names.add(line)
    return names


def catalog(model: Path, config: Path) -> list[dict[str, str]]:
    invariants = configured_invariants(config)
    entries: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_controls: set[str] = set()
    for raw_line in model.read_text(encoding="utf-8").splitlines():
        match = ANNOTATION.fullmatch(raw_line.strip())
        if not match:
            continue
        entry = match.groupdict()
        if entry["id"] in seen_ids:
            raise CatalogError(f"duplicate sabotage id: {entry['id']}")
        if entry["control"] in seen_controls:
            raise CatalogError(f"duplicate sabotage control: {entry['control']}")
        if entry["invariant"] not in invariants:
            raise CatalogError(
                f"sabotage {entry['id']} names an unchecked invariant: "
                f"{entry['invariant']}"
            )
        seen_ids.add(entry["id"])
        seen_controls.add(entry["control"])
        entries.append(entry)
    if not entries:
        raise CatalogError("TLA+ model declares no sabotage controls")
    return entries


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sounio-fleet-tla-sabotage")
    root.add_argument("--model", type=Path, required=True)
    root.add_argument("--config", type=Path, required=True)
    root.add_argument("--check-test", type=Path, action="append", default=[])
    return root


def main() -> int:
    args = parser().parse_args()
    entries = catalog(args.model, args.config)
    if args.check_test:
        test_source = "\n".join(
            path.read_text(encoding="utf-8") for path in args.check_test
        )
        missing = [
            entry["control"]
            for entry in entries
            if f"MODEL_CONTROL:{entry['control']}" not in test_source
        ]
        if missing:
            raise CatalogError(
                "model sabotage controls have no executable witness: "
                + ",".join(missing)
            )
    json.dump(
        {"model": args.model.name, "sabotages": entries, "version": 1},
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CatalogError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
