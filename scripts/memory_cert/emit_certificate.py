#!/usr/bin/env python3
"""Emit a Memory Certificate v1 JSON for a Madaros generation.

Work-novelty Spine A (RFC-0001): certificates are the object the next
generation verifies. Gen 0 starts as `Estimated` / declared arena sizes;
later gens tighten to `Exact` when proofs exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_PHASES = ("Parse", "Check", "Lower", "Codegen")

# Current Madaros fixed arenas (ADR-010 / runtime): 2 GiB heap shared.
# Phase split is a declared budget until measurement lands (RFC Stage 1).
DEFAULT_BOUNDS = {
    "Parse": 512 * 1024 * 1024,
    "Check": 1024 * 1024 * 1024,
    "Lower": 1024 * 1024 * 1024,
    "Codegen": 512 * 1024 * 1024,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_tree(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_bounds(path: Path | None) -> dict[str, int]:
    if path is None:
        return dict(DEFAULT_BOUNDS)
    data = json.loads(path.read_text())
    phases = data.get("phases") or data
    out: dict[str, int] = {}
    for name in REQUIRED_PHASES:
        entry = phases[name] if isinstance(phases, dict) else None
        if entry is None:
            raise SystemExit(f"bounds file missing phase {name}")
        if isinstance(entry, int):
            out[name] = entry
        else:
            out[name] = int(entry["bound_bytes"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--elf", type=Path, default=None, help="Compiler ELF to hash")
    ap.add_argument("--bounds", type=Path, default=None, help="JSON phase bounds override")
    ap.add_argument(
        "--kind",
        choices=("Exact", "UpperBound", "Estimated", "Unknown"),
        default="Estimated",
    )
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    bounds = load_bounds(args.bounds)
    elf_hash = sha256_file(args.elf) if args.elf and args.elf.is_file() else ""
    cert = {
        "version": 1,
        "generation": args.generation,
        "kind": args.kind.lower() if args.kind != "Unknown" else "unknown",
        "source_tree": git_tree(args.repo.resolve()),
        "elf_sha256": elf_hash,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phases": [
            {
                "id": name,
                "bound_bytes": bounds[name],
                "bound_kind": args.kind,
                "confidence": args.confidence,
                "provenance": (
                    "declared_phase_budget_v1"
                    if args.bounds is None
                    else f"bounds_file:{args.bounds.name}"
                ),
                "measured_peak_bytes": None,
            }
            for name in REQUIRED_PHASES
        ],
        "notes": args.notes
        or (
            "Gen 0 seed certificate: phase budgets declared from current "
            "fixed-arena split; not Lean-proved."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cert, indent=2) + "\n")
    print(f"emitted {args.out} generation={args.generation} elf_sha256={elf_hash or '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
