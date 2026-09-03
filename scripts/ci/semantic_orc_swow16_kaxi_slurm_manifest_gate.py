#!/usr/bin/env python3
"""Aggregate four SWOW16 K-AXI Slurm runtime results into one manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_REGIMES = ("normative", "anxious", "ruminative", "psychotic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("results", type=Path, nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    by_regime: dict[str, dict[str, object]] = {}
    for path in args.results:
        doc = json.loads(path.read_text())
        regime = str(doc.get("regime", ""))
        if regime in by_regime:
            raise SystemExit(f"duplicate regime result: {regime}")
        if doc.get("schema") != "sounio.semantic_orc.swow16_kaxi_slurm_result.v1":
            raise SystemExit(f"unexpected schema in {path}")
        if doc.get("status") != "pass":
            raise SystemExit(f"non-pass result in {path}: {doc.get('status')}")
        runtime = doc.get("runtime") or {}
        semantic = runtime.get("semantic") or {}
        tolerance = float(semantic.get("tolerance", 0.0))
        maxdu = float(semantic.get("maxdu_vs_pack_oracle", 1.0))
        maxdv = float(semantic.get("maxdv_vs_pack_oracle", 1.0))
        if maxdu > tolerance or maxdv > tolerance:
            raise SystemExit(f"oracle tolerance failure in {path}")
        by_regime[regime] = {
            "path": str(path),
            "job_id": (doc.get("job") or {}).get("id", ""),
            "maxdu_vs_pack_oracle": maxdu,
            "maxdv_vs_pack_oracle": maxdv,
            "tolerance": tolerance,
            "runtime_reason": (runtime.get("runtime") or {}).get("reason", ""),
        }

    missing = [regime for regime in EXPECTED_REGIMES if regime not in by_regime]
    if missing:
        raise SystemExit(f"missing regime results: {', '.join(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema": "sounio.semantic_orc.swow16_kaxi_slurm_manifest.v1",
                "status": "pass",
                "regimes": by_regime,
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
    print(f"semantic_orc_swow16_kaxi_slurm_manifest_gate: PASS artifact={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
