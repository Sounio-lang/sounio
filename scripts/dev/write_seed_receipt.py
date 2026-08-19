#!/usr/bin/env python3
"""Emit a SeedReceipt for a lean_single seed refresh.

A procedure says work ran. A receipt lets someone verify it was done *well*
months later without re-running the chain.

Fixed point is a *field*, not a narrative step:
  fixed_point.gk_md5 and fixed_point.gk_plus1_md5 must be identical by eye.

Usage:
  python3 scripts/dev/write_seed_receipt.py \\
    --out-dir artifacts/seed-refresh \\
    --gens-tsv path/to/gens.tsv \\
    --source path \\
    --input-seed path \\
    --output-seed path \\
    --settle-k N \\
    --placement slurm|local-locked \\
    [--slurm-partition P] [--slurm-time T] [--slurm-job-id J] [--hostname H] \\
    [--git-commit C] [--git-branch B] \\
    [--canonical-gate pass|fail|skip] [--verify-lean-seed pass|fail|skip] \\
    [--schema-version 1]

gens.tsv columns (tab-separated, no header required if 4 cols):
  gen  md5  sha256  path
  gen may be g0, g1, … or 0, 1, …
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_gens_tsv(path: Path) -> list[dict]:
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8")
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise SystemExit(f"gens.tsv line {line_no}: need gen md5 sha256 [path], got {parts!r}")
        gen_raw, md5, sha = parts[0].strip(), parts[1].strip(), parts[2].strip()
        p = parts[3].strip() if len(parts) > 3 else ""
        gen_label = gen_raw if gen_raw.startswith("g") else f"g{gen_raw}"
        try:
            gen_index = int(gen_label[1:])
        except ValueError as e:
            raise SystemExit(f"gens.tsv line {line_no}: bad gen {gen_raw!r}") from e
        rows.append(
            {
                "gen": gen_label,
                "gen_index": gen_index,
                "md5": md5.lower(),
                "sha256": sha.lower(),
                "path": p,
            }
        )
    rows.sort(key=lambda r: r["gen_index"])
    return rows


def build_fixed_point(gens: list[dict], settle_k: int) -> dict:
    """settle_k is the index k such that g_k == g_{k+1} (both present)."""
    by_i = {g["gen_index"]: g for g in gens}
    if settle_k not in by_i or (settle_k + 1) not in by_i:
        raise SystemExit(
            f"settle-k={settle_k} requires g{settle_k} and g{settle_k + 1} in gens.tsv; "
            f"have {[g['gen'] for g in gens]}"
        )
    gk = by_i[settle_k]
    gk1 = by_i[settle_k + 1]
    equal_md5 = gk["md5"] == gk1["md5"]
    equal_sha = gk["sha256"] == gk1["sha256"]
    verified = equal_md5 and equal_sha and bool(gk["md5"]) and bool(gk["sha256"])
    return {
        # Field, not a step log: a reader must confirm equality by eye.
        "criterion": "md5(g_k) == md5(g_{k+1}) AND sha256(g_k) == sha256(g_{k+1})",
        "k": settle_k,
        "k_plus_1": settle_k + 1,
        "gk_label": gk["gen"],
        "gk_plus1_label": gk1["gen"],
        "gk_md5": gk["md5"],
        "gk_plus1_md5": gk1["md5"],
        "gk_sha256": gk["sha256"],
        "gk_plus1_sha256": gk1["sha256"],
        "md5_equal": equal_md5,
        "sha256_equal": equal_sha,
        "verified": verified,
        # Side-by-side block for human eyeball (duplicated in .txt):
        "eyeball_md5": f"{gk['md5']}\n{gk1['md5']}",
        "eyeball_sha256": f"{gk['sha256']}\n{gk1['sha256']}",
    }


def human_text(receipt: dict) -> str:
    fp = receipt["fixed_point"]
    env = receipt["environment"]
    lines = [
        "SeedReceipt",
        f"schema_version: {receipt['schema_version']}",
        f"receipt_utc: {receipt['receipt_utc']}",
        f"git_commit: {receipt.get('git_commit') or ''}",
        f"git_branch: {receipt.get('git_branch') or ''}",
        "",
        "## source (lean_single.sio)",
        f"path: {receipt['source']['path']}",
        f"sha256: {receipt['source']['sha256']}",
        f"md5: {receipt['source']['md5']}",
        f"bytes: {receipt['source']['bytes']}",
        "",
        "## input seed (g0 / bootstrap ELF used to start the chain)",
        f"path: {receipt['input_seed']['path']}",
        f"sha256: {receipt['input_seed']['sha256']}",
        f"md5: {receipt['input_seed']['md5']}",
        f"bytes: {receipt['input_seed']['bytes']}",
        "",
        "## generations",
    ]
    for g in receipt["generations"]:
        lines.append(
            f"  {g['gen']}: md5={g['md5']}  sha256={g['sha256']}  path={g.get('path') or ''}"
        )
    lines += [
        "",
        "## fixed_point (FIELD — confirm equality by eye; if you cannot, receipt proves nothing)",
        f"criterion: {fp['criterion']}",
        f"k: {fp['k']}",
        f"k_plus_1: {fp['k_plus_1']}",
        f"gk_label: {fp['gk_label']}",
        f"gk_plus1_label: {fp['gk_plus1_label']}",
        "--- md5 side-by-side (must be identical) ---",
        f"gk_md5:       {fp['gk_md5']}",
        f"gk_plus1_md5: {fp['gk_plus1_md5']}",
        f"md5_equal: {str(fp['md5_equal']).lower()}",
        "--- sha256 side-by-side (must be identical) ---",
        f"gk_sha256:       {fp['gk_sha256']}",
        f"gk_plus1_sha256: {fp['gk_plus1_sha256']}",
        f"sha256_equal: {str(fp['sha256_equal']).lower()}",
        f"verified: {str(fp['verified']).lower()}",
        "",
        "## output seed (installed / to commit)",
        f"path: {receipt['output_seed']['path']}",
        f"sha256: {receipt['output_seed']['sha256']}",
        f"md5: {receipt['output_seed']['md5']}",
        f"bytes: {receipt['output_seed']['bytes']}",
        "",
        "## environment",
        f"placement: {env.get('placement') or ''}",
        f"hostname: {env.get('hostname') or ''}",
        f"slurm_partition: {env.get('slurm_partition') or ''}",
        f"slurm_time: {env.get('slurm_time') or ''}",
        f"slurm_job_id: {env.get('slurm_job_id') or ''}",
        f"slurm_nodelist: {env.get('slurm_nodelist') or ''}",
        "",
        "## post checks",
        f"canonical_compiler_gate: {receipt['checks'].get('canonical_compiler_gate')}",
        f"verify_lean_seed: {receipt['checks'].get('verify_lean_seed')}",
        "",
        "## what this receipt does NOT prove",
        receipt["limits"]["provenance_note"],
        "",
    ]
    return "\n".join(lines)


def verify_receipt_file(path: Path) -> int:
    """Exit 0 if fixed_point fields match by equality; 1 otherwise."""
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = receipt_internal_errors(data)
    if errors:
        print(f"[seed-receipt] FAIL {path}", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    fp = data.get("fixed_point") or {}
    print(f"[seed-receipt] PASS fixed_point verified by eye fields in {path}")
    print(f"  gk_md5:       {fp.get('gk_md5')}")
    print(f"  gk_plus1_md5: {fp.get('gk_plus1_md5')}")
    return 0


def receipt_internal_errors(data: dict) -> list[str]:
    """Structural / fixed-point field checks (no tree)."""
    errors: list[str] = []
    if data.get("schema") not in (None, "sounio.SeedReceipt"):
        # allow missing schema on older drafts; if present must match
        if data.get("schema") != "sounio.SeedReceipt":
            errors.append(f"schema must be sounio.SeedReceipt, got {data.get('schema')!r}")
    fp = data.get("fixed_point") or {}
    if not fp:
        errors.append("missing fixed_point object")
        return errors
    if not fp.get("verified"):
        errors.append("fixed_point.verified is not true")
    gk_md5 = (fp.get("gk_md5") or "").lower()
    gk1_md5 = (fp.get("gk_plus1_md5") or "").lower()
    gk_sha = (fp.get("gk_sha256") or "").lower()
    gk1_sha = (fp.get("gk_plus1_sha256") or "").lower()
    if not gk_md5 or not gk1_md5:
        errors.append("fixed_point missing gk_md5 / gk_plus1_md5 (must be written side by side)")
    elif gk_md5 != gk1_md5:
        errors.append(
            "md5 mismatch by eye (provenance refuse):\n"
            f"  gk_md5:       {gk_md5}\n"
            f"  gk_plus1_md5: {gk1_md5}"
        )
    if not gk_sha or not gk1_sha:
        errors.append("fixed_point missing gk_sha256 / gk_plus1_sha256")
    elif gk_sha != gk1_sha:
        errors.append(
            "sha256 mismatch by eye:\n"
            f"  gk_sha256:       {gk_sha}\n"
            f"  gk_plus1_sha256: {gk1_sha}"
        )
    gens = data.get("generations") or []
    if len(gens) < 2:
        errors.append("generations[] must list at least g_k and g_{k+1} (need ≥2 entries)")
    # settled hashes must appear in the generation table
    md5s = {(g.get("md5") or "").lower() for g in gens}
    if gk_md5 and gk_md5 not in md5s:
        errors.append(f"fixed_point.gk_md5 {gk_md5} not present in generations[]")
    src = data.get("source") or {}
    if not (src.get("sha256") or src.get("md5")):
        errors.append("source missing sha256/md5")
    out = data.get("output_seed") or {}
    if not (out.get("sha256") or out.get("md5")):
        errors.append("output_seed missing sha256/md5")
    inn = data.get("input_seed") or {}
    if not (inn.get("sha256") or inn.get("md5")):
        errors.append("input_seed missing sha256/md5 (g0 of the chain)")
    return errors


def check_receipt_against_tree(
    receipt_path: Path,
    *,
    source_path: Path,
    seed_path: Path,
) -> list[str]:
    """
    Provenance check: receipt must describe *this* tree's source + committed ELF.

    Self-repro alone (canonical_compiler_gate) only proves the ELF reproduces
    itself on the source. This check proves the receipt's claimed source SHA
    matches the committed lean_single.sio and the claimed output matches the
    committed seed ELF — so a substituted foreign fixed-point ELF with a
    mismatched paper trail fails.
    """
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    errors = receipt_internal_errors(data)
    if not source_path.is_file():
        errors.append(f"source file missing: {source_path}")
        return errors
    if not seed_path.is_file():
        errors.append(f"seed ELF missing: {seed_path}")
        return errors

    live_src_sha = sha256_file(source_path)
    live_src_md5 = md5_file(source_path)
    live_seed_sha = sha256_file(seed_path)
    live_seed_md5 = md5_file(seed_path)

    claimed_src_sha = ((data.get("source") or {}).get("sha256") or "").lower()
    claimed_src_md5 = ((data.get("source") or {}).get("md5") or "").lower()
    claimed_out_sha = ((data.get("output_seed") or {}).get("sha256") or "").lower()
    claimed_out_md5 = ((data.get("output_seed") or {}).get("md5") or "").lower()

    if claimed_src_sha and claimed_src_sha != live_src_sha:
        errors.append(
            "PROVENANCE FAIL: receipt source.sha256 does not match committed lean_single.sio\n"
            f"  receipt:   {claimed_src_sha}\n"
            f"  committed: {live_src_sha}\n"
            "  (A receipt that names another source cannot vouch for this tree.)"
        )
    elif claimed_src_md5 and not claimed_src_sha and claimed_src_md5 != live_src_md5:
        errors.append(
            "PROVENANCE FAIL: receipt source.md5 does not match committed lean_single.sio\n"
            f"  receipt:   {claimed_src_md5}\n"
            f"  committed: {live_src_md5}"
        )
    elif not claimed_src_sha and not claimed_src_md5:
        errors.append("receipt source has neither sha256 nor md5")

    if claimed_out_sha and claimed_out_sha != live_seed_sha:
        errors.append(
            "PROVENANCE FAIL: receipt output_seed.sha256 does not match committed seed ELF\n"
            f"  receipt:   {claimed_out_sha}\n"
            f"  committed: {live_seed_sha}"
        )
    elif claimed_out_md5 and not claimed_out_sha and claimed_out_md5 != live_seed_md5:
        errors.append(
            "PROVENANCE FAIL: receipt output_seed.md5 does not match committed seed ELF\n"
            f"  receipt:   {claimed_out_md5}\n"
            f"  committed: {live_seed_md5}"
        )
    elif not claimed_out_sha and not claimed_out_md5:
        errors.append("receipt output_seed has neither sha256 nor md5")

    # Settled generation must match the committed ELF
    fp = data.get("fixed_point") or {}
    gk_md5 = (fp.get("gk_md5") or "").lower()
    if gk_md5 and gk_md5 != live_seed_md5:
        errors.append(
            "PROVENANCE FAIL: fixed_point.gk_md5 does not match committed seed ELF md5\n"
            f"  gk_md5:    {gk_md5}\n"
            f"  seed md5:  {live_seed_md5}"
        )
    gk_sha = (fp.get("gk_sha256") or "").lower()
    if gk_sha and gk_sha != live_seed_sha:
        errors.append(
            "PROVENANCE FAIL: fixed_point.gk_sha256 does not match committed seed ELF sha256\n"
            f"  gk_sha256:   {gk_sha}\n"
            f"  seed sha256: {live_seed_sha}"
        )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-receipt", metavar="PATH", help="validate fixed_point fields only")
    ap.add_argument(
        "--check-against-tree",
        metavar="RECEIPT",
        help="validate receipt against --source and --seed-elf on disk (provenance)",
    )
    ap.add_argument("--seed-elf", type=Path, help="committed lean_single ELF (with --check-against-tree)")
    ap.add_argument("--out-dir", type=Path, help="directory for SeedReceipt-*.json and .txt")
    ap.add_argument("--gens-tsv", type=Path, help="generation table")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--input-seed", type=Path)
    ap.add_argument("--output-seed", type=Path)
    ap.add_argument("--settle-k", type=int, help="k where g_k == g_{k+1}")
    ap.add_argument("--placement", choices=("slurm", "local-locked", "check-only", "unknown"), default="unknown")
    ap.add_argument("--slurm-partition", default="")
    ap.add_argument("--slurm-time", default="")
    ap.add_argument("--slurm-job-id", default="")
    ap.add_argument("--slurm-nodelist", default="")
    ap.add_argument("--hostname", default="")
    ap.add_argument("--git-commit", default="")
    ap.add_argument("--git-branch", default="")
    ap.add_argument("--canonical-gate", default="skip", choices=("pass", "fail", "skip"))
    ap.add_argument("--verify-lean-seed", default="skip", choices=("pass", "fail", "skip"))
    ap.add_argument("--schema-version", type=int, default=SCHEMA_VERSION)
    args = ap.parse_args()

    if args.verify_receipt:
        return verify_receipt_file(Path(args.verify_receipt))

    if args.check_against_tree:
        receipt = Path(args.check_against_tree)
        src = args.source
        seed = args.seed_elf or args.output_seed
        if src is None or seed is None:
            ap.error("--check-against-tree requires --source and --seed-elf (or --output-seed)")
        errs = check_receipt_against_tree(receipt, source_path=src.resolve(), seed_path=seed.resolve())
        if errs:
            print(f"[seed-receipt] FAIL provenance check {receipt}", file=sys.stderr)
            for e in errs:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print(f"[seed-receipt] PASS provenance: receipt matches source+seed on disk")
        print(f"  source={src}")
        print(f"  seed={seed}")
        return 0

    required = [
        args.out_dir,
        args.gens_tsv,
        args.source,
        args.input_seed,
        args.output_seed,
        args.settle_k,
    ]
    if any(x is None for x in required):
        ap.error("emit mode requires --out-dir --gens-tsv --source --input-seed --output-seed --settle-k")

    source = args.source.resolve()
    in_seed = args.input_seed.resolve()
    out_seed = args.output_seed.resolve()
    for p in (source, in_seed, out_seed, args.gens_tsv):
        if not p.is_file():
            raise SystemExit(f"missing file: {p}")

    gens = parse_gens_tsv(args.gens_tsv)
    fp = build_fixed_point(gens, args.settle_k)
    if not fp["verified"]:
        raise SystemExit(
            "refusing to emit receipt: fixed_point not verified "
            f"(md5_equal={fp['md5_equal']} sha256_equal={fp['sha256_equal']})"
        )

    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    receipt = {
        "schema": "sounio.SeedReceipt",
        "schema_version": args.schema_version,
        "receipt_utc": utc,
        "git_commit": args.git_commit,
        "git_branch": args.git_branch,
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "md5": md5_file(source),
            "bytes": source.stat().st_size,
        },
        "input_seed": {
            "path": str(in_seed),
            "sha256": sha256_file(in_seed),
            "md5": md5_file(in_seed),
            "bytes": in_seed.stat().st_size,
            "role": "g0_start_of_chain",
        },
        "generations": gens,
        "fixed_point": fp,
        "output_seed": {
            "path": str(out_seed),
            "sha256": sha256_file(out_seed),
            "md5": md5_file(out_seed),
            "bytes": out_seed.stat().st_size,
        },
        "environment": {
            "placement": args.placement,
            "hostname": args.hostname or os.uname().nodename,
            "slurm_partition": args.slurm_partition,
            "slurm_time": args.slurm_time,
            "slurm_job_id": args.slurm_job_id or os.environ.get("SLURM_JOB_ID", ""),
            "slurm_nodelist": args.slurm_nodelist or os.environ.get("SLURM_NODELIST", ""),
        },
        "checks": {
            "canonical_compiler_gate": args.canonical_gate,
            "verify_lean_seed": args.verify_lean_seed,
        },
        "limits": {
            "provenance_note": (
                "This receipt proves the installed ELF is a self-reproducing fixed point "
                "of the recorded source bytes (g_k == g_{k+1}, self-repro). It does NOT by "
                "itself prove the ELF was *derived from* that source rather than substituted "
                "from another generation that happens to self-reproduce the same source. "
                "canonical_compiler_gate.sh checks md5(committed)==md5(self-compile) — "
                "stability, not provenance. Provenance (ELF came FROM this source via this "
                "chain) is what the generation table + input_seed hashes are for; a future "
                "gate should require a SeedReceipt, not only the self-repro equality."
            )
        },
    }

    # Consistency: output seed must match settled generation hash
    settled_md5 = fp["gk_md5"]
    if receipt["output_seed"]["md5"] != settled_md5:
        raise SystemExit(
            f"output seed md5 {receipt['output_seed']['md5']} != settled gk_md5 {settled_md5}"
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"SeedReceipt-{stamp}"
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")
    json_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    txt_path.write_text(human_text(receipt), encoding="utf-8")

    # Also write a stable "latest" pointer in the out dir when under stage/out
    latest_json = out_dir / "SeedReceipt.latest.json"
    latest_txt = out_dir / "SeedReceipt.latest.txt"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_txt.write_text(txt_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[seed-receipt] wrote {json_path}")
    print(f"[seed-receipt] wrote {txt_path}")
    print(f"[seed-receipt] fixed_point.verified={fp['verified']}")
    print(f"[seed-receipt] gk_md5:       {fp['gk_md5']}")
    print(f"[seed-receipt] gk_plus1_md5: {fp['gk_plus1_md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
