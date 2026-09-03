#!/usr/bin/env python3
"""Aggregate SWOW16 K-AXI Slurm runtime results across regimes and supports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_REGIMES = ("normative", "anxious", "ruminative", "psychotic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--support-count", type=int, default=4)
    parser.add_argument("--support-start", type=int, default=0)
    parser.add_argument("results", type=Path, nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.support_count <= 0:
        raise SystemExit("--support-count must be positive")
    if args.support_start < 0:
        raise SystemExit("--support-start must be non-negative")

    expected_supports = tuple(range(args.support_start, args.support_start + args.support_count))
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    for path in args.results:
        doc = json.loads(path.read_text())
        if doc.get("schema") != "sounio.semantic_orc.swow16_kaxi_slurm_result.v1":
            raise SystemExit(f"unexpected schema in {path}")
        if doc.get("status") != "pass":
            raise SystemExit(f"non-pass result in {path}: {doc.get('status')}")
        regime = str(doc.get("regime", ""))
        if regime not in EXPECTED_REGIMES:
            raise SystemExit(f"unexpected regime in {path}: {regime}")
        support_index = int(doc.get("support_index", 0))
        key = (regime, support_index)
        if key in by_key:
            raise SystemExit(f"duplicate regime/support result: {regime}/{support_index}")
        runtime = doc.get("runtime") or {}
        semantic = runtime.get("semantic") or {}
        tolerance = float(semantic.get("tolerance", 0.0))
        maxdu = float(semantic.get("maxdu_vs_pack_oracle", 1.0))
        maxdv = float(semantic.get("maxdv_vs_pack_oracle", 1.0))
        if maxdu > tolerance or maxdv > tolerance:
            raise SystemExit(f"oracle tolerance failure in {path}")
        by_key[key] = {
            "path": str(path),
            "job_id": (doc.get("job") or {}).get("id", ""),
            "support_offset": doc.get("support_offset", runtime.get("support_offset", 0)),
            "maxdu_vs_pack_oracle": maxdu,
            "maxdv_vs_pack_oracle": maxdv,
            "tolerance": tolerance,
            "runtime_reason": (runtime.get("runtime") or {}).get("reason", ""),
        }

    missing = [
        (regime, support_index)
        for regime in EXPECTED_REGIMES
        for support_index in expected_supports
        if (regime, support_index) not in by_key
    ]
    if missing:
        raise SystemExit(f"missing regime/support results: {missing}")

    regimes: dict[str, dict[str, object]] = {}
    for regime in EXPECTED_REGIMES:
        regimes[regime] = {
            str(support_index): by_key[(regime, support_index)]
            for support_index in expected_supports
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema": "sounio.semantic_orc.swow16_kaxi_slurm_multisupport_manifest.v1",
                "status": "pass",
                "support_start": args.support_start,
                "support_count_per_regime": args.support_count,
                "runtime_result_count": len(by_key),
                "regimes": regimes,
                "boundaries": [
                    "gpu_runtime_compared_to_pack_oracle_only",
                    "entropic_regularized_transport_only",
                    "clinical_claims_not_enforced",
                    "no_convergence_theorem",
                    "not_a_depression_biomarker_validation",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"semantic_orc_swow16_kaxi_slurm_multisupport_manifest_gate: PASS artifact={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
