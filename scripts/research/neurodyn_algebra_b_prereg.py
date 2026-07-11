#!/usr/bin/env python3
"""Build the preregistered Algebra-B synthetic non-associativity package.

The package is intentionally synthetic and non-clinical. It freezes the next
scientific question before any larger run: does octonionic non-associativity do
work that an associative 8-D, parameter-matched control cannot do?
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.algebra_b_prereg.v1"
DECISION_TABLE = {
    "route_1_algebraic_necessity": (
        "O-SSM >=55%, A8-SSM <55%, associative projection <55%, and H-SSM is "
        "outside the 3.0 pp matching margin. Only this route permits 99 nulls."
    ),
    "route_2_dimensionality_not_algebra": (
        "O-SSM >=55% and A8-SSM >=55%, or associative projection >=55%, or H-SSM "
        "matches within 3.0 pp. Reframe as dimensionality/capacity, not octonionic necessity."
    ),
    "route_3_subthreshold_reformulation": (
        "O-SSM <55% before the reformulation budget is exhausted. A new objective/training "
        "reformulation is allowed only under a new preregistration with fixed seed/threshold."
    ),
    "route_4_terminal_negative": (
        "Two reformulations are exhausted while O-SSM remains <55%. Terminate this fixed-dim6 design."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026070801)
    parser.add_argument("--pairs", type=int, default=28)
    parser.add_argument("--sites", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--null-count", type=int, default=99)
    parser.add_argument("--max-reformulations", type=int, default=2)
    parser.add_argument("--target-assoc-dim", type=int, default=6)
    parser.add_argument("--target-assoc-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--triple-source", choices=("scaled_unit", "continuous", "unit"), default="scaled_unit")
    parser.add_argument("--scale-jitter", type=float, default=0.12)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--manifest-script", type=Path, default=Path("scripts/research/neurodyn_octonionic_associator_manifest.py"))
    parser.add_argument("--audit-script", type=Path, default=Path("scripts/research/neurodyn_octonionic_associator_data_audit.py"))
    parser.add_argument("--balance-script", type=Path, default=Path("scripts/research/neurodyn_associator_manifest_balance_gate.py"))
    parser.add_argument("--null-script", type=Path, default=Path("scripts/research/neurodyn_pair_label_permutation_manifest.py"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Algebra-B Preregistration: Necessity of Non-Associativity",
        "",
        "Claim boundary: synthetic non-clinical algebra-necessity assay only. "
        "No clinical, biomarker, biological mechanism, treatment-response, or broad O-SSM superiority claim.",
        "",
        "## Locked Question",
        "",
        "Does octonionic non-associative composition carry a third-order signal that a "
        "parameter-matched associative 8-D control cannot recover under the same shortcuts-off training surface?",
        "",
        "## Design",
        "",
        f"- True manifest pairs/sites/seq_len: `{payload['pairs']}` / `{payload['sites']}` / `{payload['seq_len']}`",
        f"- Triple source: `{payload['triple_source']}`",
        f"- Target associator component/sign: `{payload['target_assoc_dim']}` / `{payload['target_assoc_sign']}`",
        f"- Null count: `{payload['null_count']}` pair-label permutations",
        "- Required shortcuts-off config: `READOUT_MEAN_SCALE=0`, `READOUT_DELTA_SCALE=0`, `READOUT_FLAT_SCALE=0`",
        "- Required baselines before null expansion: O-SSM, H-SSM, A8-SSM (`H+H` direct-sum associative 8-D), associative-projection O-SSM.",
        f"- Reformulation budget for this fixed-dim6 line: `{payload['max_reformulations']}`.",
        "",
        "## Decision Table",
        "",
    ]
    for key, value in DECISION_TABLE.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- True manifest package: `{payload['true_manifest_dir']}`",
            f"- Data audit: `{payload['audit_dir']}`",
            f"- Balance gate: `{payload['balance_dir']}`",
            f"- Null manifest root: `{payload['null_manifest_root']}`",
            f"- Runbook: `{payload['runbook']}`",
            "",
            "## Stop Rule",
            "",
            "Run the O/H/A8/projection attribution smoke before nulls. If O-SSM is below 55%, stop and consume one "
            "reformulation attempt. If A8 or projection crosses 55%, stop and reframe as non-octonionic. "
            "Only route 1 may run 99 nulls.",
            "",
        ]
    )
    return "\n".join(lines)


def runbook(payload: dict[str, Any]) -> str:
    manifest = Path(payload["true_manifest_dir"]) / "octonionic_associator_manifest.tsv"
    lines = [
        "# Algebra-B Slurm Runbook",
        "",
        "This runbook is executable guidance, not evidence. Evidence begins only after SHA-checked Slurm outputs exist.",
        "",
        "## True Smoke",
        "",
        "```bash",
        "RUN_ID=neurodyn-algebra-b-true-smoke \\",
        f"MANIFEST_PATH={manifest} \\",
        "OUTPUT_DIR=artifacts/research/neurodyn/synthetic/algebra_b_true_smoke_$(date -u +%Y%m%dT%H%M%SZ) \\",
        "PAIRS_EXPECTED=" + str(payload["pairs"]) + " \\",
        "NODE=gpuorangefs-5860-proxmox \\",
        "READOUT_MEAN_SCALE=0 READOUT_DELTA_SCALE=0 READOUT_FLAT_SCALE=0 \\",
        "TRACE_HIDDEN_STATE=1 TRACE_READOUT_ALL_FOLDS=1 \\",
        "bash scripts/research/neurodyn_direct_slurm_smoke.sh",
        "```",
        "",
        "## Bridge Nulls",
        "",
        "Run null seeds 1..5 first using the manifests under `null_manifests/`. Do not run nulls 6..99 until the gate clears.",
        "",
        "## Required Missing Baselines",
        "",
        "The decision gate requires A8-SSM and associative-projection O-SSM outputs. Until those model surfaces are wired, "
        "the final gate must report `ALGEBRA_B_NOT_READY_MISSING_ASSOCIATIVE_CONTROLS`, even if O-SSM beats H-SSM/nulls.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    true_dir = out / "true_manifest"
    audit_dir = out / "data_audit"
    balance_dir = out / "balance_gate"
    null_root = out / "null_manifests"
    null_root.mkdir(parents=True, exist_ok=True)

    manifest_cmd = [
        sys.executable,
        str(args.manifest_script),
        "--output-dir",
        str(true_dir),
        "--pairs",
        str(args.pairs),
        "--sites",
        str(args.sites),
        "--seq-len",
        str(args.seq_len),
        "--seed",
        str(args.seed),
        "--noise-std",
        str(args.noise_std),
        "--site-mode",
        "round_robin",
        "--triple-selection",
        "seeded_shuffle",
        "--triple-source",
        args.triple_source,
        "--target-assoc-dim",
        str(args.target_assoc_dim),
        "--target-assoc-sign",
        str(args.target_assoc_sign),
        "--scale-jitter",
        str(args.scale_jitter),
        "--overwrite",
    ]
    run_cmd(manifest_cmd)

    manifest = true_dir / "octonionic_associator_manifest.tsv"
    triples = true_dir / "associator_triples.tsv"
    run_cmd([sys.executable, str(args.audit_script), "--manifest", str(manifest), "--output-dir", str(audit_dir), "--overwrite"])
    run_cmd(
        [
            sys.executable,
            str(args.balance_script),
            "--manifest",
            str(manifest),
            "--associator-triples",
            str(triples),
            "--output-dir",
            str(balance_dir),
            "--overwrite",
        ]
    )

    null_dirs: list[str] = []
    for idx in range(1, args.null_count + 1):
        null_seed = args.seed + 10_000 + idx
        null_dir = null_root / f"pairpermnull_{idx:03d}_seed{null_seed}"
        run_cmd(
            [
                sys.executable,
                str(args.null_script),
                "--input",
                str(manifest),
                "--output-dir",
                str(null_dir),
                "--seed",
                str(null_seed),
                "--overwrite",
            ]
        )
        null_dirs.append(rel(null_dir))

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "pairs": args.pairs,
        "sites": args.sites,
        "seq_len": args.seq_len,
        "null_count": args.null_count,
        "triple_source": args.triple_source,
        "target_assoc_dim": args.target_assoc_dim,
        "target_assoc_sign": args.target_assoc_sign,
        "claim_boundary": "Synthetic non-clinical algebra-necessity assay only.",
        "decision_table": DECISION_TABLE,
        "true_manifest_dir": rel(true_dir),
        "audit_dir": rel(audit_dir),
        "balance_dir": rel(balance_dir),
        "null_manifest_root": rel(null_root),
        "null_manifest_dirs": null_dirs,
        "runbook": rel(out / "runbook.md"),
        "required_controls": [
            "O-SSM true and null outputs",
            "H-SSM true and null outputs",
            "A8-SSM direct-sum associative 8-D true and null outputs",
            "associative-projection O-SSM true and null outputs",
        ],
        "max_reformulations": args.max_reformulations,
        "stop_rule": DECISION_TABLE["route_3_subthreshold_reformulation"],
    }
    write_json(out / "algebra_b_prereg.json", payload)
    (out / "algebra_b_prereg.md").write_text(markdown(payload), encoding="utf-8")
    (out / "runbook.md").write_text(runbook(payload), encoding="utf-8")

    with (out / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
