#!/usr/bin/env python3
"""Verify a Memory Certificate v1 (and optional fixed-point against another)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REQUIRED_PHASES = ("Parse", "Check", "Lower", "Codegen")
VALID_KINDS = frozenset({"Exact", "UpperBound", "Estimated", "Unknown"})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def check_cert(cert: dict, *, allow_unknown: bool, elf: Path | None) -> list[str]:
    errs: list[str] = []
    if cert.get("version") != 1:
        errs.append(f"version want 1 got {cert.get('version')!r}")
    phases = cert.get("phases")
    if not isinstance(phases, list) or not phases:
        errs.append("phases missing or empty")
        return errs
    by_id = {p.get("id"): p for p in phases if isinstance(p, dict)}
    for name in REQUIRED_PHASES:
        if name not in by_id:
            errs.append(f"missing phase {name}")
            continue
        p = by_id[name]
        b = p.get("bound_bytes")
        if not isinstance(b, int) or b < 0:
            errs.append(f"phase {name}: bad bound_bytes {b!r}")
        kind = p.get("bound_kind")
        if kind not in VALID_KINDS:
            errs.append(f"phase {name}: bad bound_kind {kind!r}")
        elif kind == "Unknown" and not allow_unknown:
            errs.append(f"phase {name}: Unknown bound refused (pass --allow-unknown)")
    elf_hash = cert.get("elf_sha256") or ""
    if elf is not None:
        if not elf.is_file():
            errs.append(f"elf not found: {elf}")
        else:
            got = sha256_file(elf)
            if not elf_hash:
                errs.append("certificate elf_sha256 empty but --elf given")
            elif got != elf_hash:
                errs.append(f"elf sha256 mismatch cert={elf_hash[:16]}… file={got[:16]}…")
    return errs


def fixed_point(a: dict, b: dict) -> list[str]:
    errs: list[str] = []
    pa = {p["id"]: p for p in a.get("phases", []) if isinstance(p, dict) and "id" in p}
    pb = {p["id"]: p for p in b.get("phases", []) if isinstance(p, dict) and "id" in p}
    if set(pa) != set(pb):
        errs.append(f"phase id sets differ: {sorted(pa)} vs {sorted(pb)}")
    for name in sorted(set(pa) & set(pb)):
        if pa[name].get("bound_bytes") != pb[name].get("bound_bytes"):
            errs.append(
                f"{name}: bound_bytes {pa[name].get('bound_bytes')} != {pb[name].get('bound_bytes')}"
            )
        if pa[name].get("bound_kind") != pb[name].get("bound_kind"):
            errs.append(
                f"{name}: bound_kind {pa[name].get('bound_kind')} != {pb[name].get('bound_kind')}"
            )
    ha, hb = a.get("elf_sha256") or "", b.get("elf_sha256") or ""
    if ha and hb and ha != hb:
        errs.append("elf_sha256 differs between certificates (not fixed point)")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cert", type=Path)
    ap.add_argument("--elf", type=Path, default=None)
    ap.add_argument("--allow-unknown", action="store_true")
    ap.add_argument(
        "--fixed-point-with",
        type=Path,
        default=None,
        help="Second certificate; require matching phase bounds + elf hash",
    )
    args = ap.parse_args()

    if not args.cert.is_file():
        print(f"FAIL: missing {args.cert}", file=sys.stderr)
        return 2
    cert = load(args.cert)
    errs = check_cert(cert, allow_unknown=args.allow_unknown, elf=args.elf)
    if args.fixed_point_with is not None:
        if not args.fixed_point_with.is_file():
            errs.append(f"missing fixed-point peer {args.fixed_point_with}")
        else:
            errs.extend(fixed_point(cert, load(args.fixed_point_with)))

    if errs:
        print("VERIFY_FAIL:")
        for e in errs:
            print(f"  - {e}")
        return 1
    print(
        f"VERIFY_OK: generation={cert.get('generation')} "
        f"phases={len(cert.get('phases', []))} "
        f"elf={'(checked)' if args.elf else '(skipped)'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
