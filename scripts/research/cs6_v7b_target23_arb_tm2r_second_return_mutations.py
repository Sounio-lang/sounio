#!/usr/bin/env python3
"""Adversarial receipt mutations for the Arb TM2R event projection."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


MUTATIONS = (
    ("worker_hash", "WORKER_SOURCE_SHA256=", "WORKER_SOURCE_SHA256=" + "0" * 64),
    ("dependency_hash", "FIRST_RETURN_DEPENDENCY_SHA256=", "FIRST_RETURN_DEPENDENCY_SHA256=" + "0" * 64),
    ("leaf", "LEAF_ID=", "LEAF_ID=U00-0000000000_S00-0000000000"),
    ("second_degree", "SECOND_PHASE_SOURCE_DEGREE=2", "SECOND_PHASE_SOURCE_DEGREE=3"),
    ("second_order", "SECOND_PHASE_TIME_TAYLOR_ORDER=12", "SECOND_PHASE_TIME_TAYLOR_ORDER=24"),
    ("projection_method", "EVENT_PROJECTION=", "EVENT_PROJECTION=POINT_NEWTON"),
    ("failure_class", "FAILURE_CLASS=", "FAILURE_CLASS=NONE"),
    ("failure_phase", "FAILURE_PHASE=", "FAILURE_PHASE=NONE"),
    ("attempted", "TOTAL_ATTEMPTED_STEPS=1325", "TOTAL_ATTEMPTED_STEPS=1324"),
    ("completed", "TOTAL_COMPLETED_STEPS=1324", "TOTAL_COMPLETED_STEPS=1325"),
    ("picard", "TOTAL_PICARD_CONTAINMENTS=1325", "TOTAL_PICARD_CONTAINMENTS=1324"),
    ("endpoint", "TOTAL_ENDPOINT_PICARD_CONTAINMENTS=1324", "TOTAL_ENDPOINT_PICARD_CONTAINMENTS=1325"),
    ("reconditionings", "TOTAL_RECONDITIONINGS=1326", "TOTAL_RECONDITIONINGS=0"),
    ("generators", "TOTAL_GENERATOR_RECONSTRUCTIONS=34213", "TOTAL_GENERATOR_RECONSTRUCTIONS=0"),
    ("first_step", "FIRST_RETURN_END_STEP=617", "FIRST_RETURN_END_STEP=616"),
    ("before_sign", "FIRST_RETURN_W_BEFORE_UPPER_Q=", "FIRST_RETURN_W_BEFORE_UPPER_Q=0"),
    ("after_sign", "FIRST_RETURN_W_AFTER_LOWER_Q=", "FIRST_RETURN_W_AFTER_LOWER_Q=0"),
    ("denominator", "NEWTON_DENOMINATOR_LOWER_Q=", "NEWTON_DENOMINATOR_LOWER_Q=0"),
    ("delta_lower", "NEWTON_TIME_CORRECTION_LOWER_Q=", "NEWTON_TIME_CORRECTION_LOWER_Q=-1/128"),
    ("fixed_interval", "NEWTON_FIXED_TIME_SHIFT_UPPER_Q=", "NEWTON_FIXED_TIME_SHIFT_UPPER_Q=0"),
    ("residual_symmetry", "NEWTON_RESIDUAL_TIME_SHIFT_UPPER_Q=", "NEWTON_RESIDUAL_TIME_SHIFT_UPPER_Q=0"),
    ("source_terms", "PURE_SOURCE_MONOMIALS_RETAINED=15", "PURE_SOURCE_MONOMIALS_RETAINED=0"),
    ("projection_contraction", "PROJECTION_PICARD_CONTRACTION_UPPER_Q=", "PROJECTION_PICARD_CONTRACTION_UPPER_Q=1"),
    ("slab_contraction", "PROJECTION_SLAB_PICARD_CONTRACTION_UPPER_Q=", "PROJECTION_SLAB_PICARD_CONTRACTION_UPPER_Q=1"),
    ("slab_containment", "PROJECTION_SLAB_CONTAINED_IN_EVENT_TUBE=true", "PROJECTION_SLAB_CONTAINED_IN_EVENT_TUBE=false"),
    ("projection_width", "PROJECTED_CARRIER_MAX_WIDTH_UPPER_Q=", "PROJECTED_CARRIER_MAX_WIDTH_UPPER_Q=1"),
    ("section", "PROJECTED_W_EXACTLY_ZERO=true", "PROJECTED_W_EXACTLY_ZERO=false"),
    ("second_attempted", "SECOND_PHASE_ATTEMPTED_STEPS=708", "SECOND_PHASE_ATTEMPTED_STEPS=707"),
    ("second_completed", "SECOND_PHASE_COMPLETED_STEPS=707", "SECOND_PHASE_COMPLETED_STEPS=708"),
    ("second_time", "SECOND_PHASE_COMPLETED_TIME_Q=707/256", "SECOND_PHASE_COMPLETED_TIME_Q=177/64"),
    ("projection_certificate", "INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE=true", "INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE=false"),
    ("second_certificate", "FULL_LEAF_SECOND_RETURN_CERTIFICATE=false", "FULL_LEAF_SECOND_RETURN_CERTIFICATE=true"),
    ("point_fallback", "POINT_FALLBACK_USED=false", "POINT_FALLBACK_USED=true"),
    ("chaos", "CHAOS_PROVED=false", "CHAOS_PROVED=true"),
    ("open_problem", "OPEN_PROBLEM_SOLVED=false", "OPEN_PROBLEM_SOLVED=true"),
)


def mutate(text: str, prefix: str, replacement: str) -> str:
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise SystemExit(f"mutation anchor population mismatch: {prefix}")
    lines[matches[0]] = replacement
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("verifier", type=Path)
    args = parser.parse_args()
    original = args.output.read_text(encoding="ascii")
    rejected = 0
    with tempfile.TemporaryDirectory(prefix="cs6-arb-tm2r-second-return-mutations-") as raw:
        directory = Path(raw)
        for index, (name, prefix, replacement) in enumerate(MUTATIONS):
            path = directory / f"{index:02d}-{name}.txt"
            path.write_text(mutate(original, prefix, replacement), encoding="ascii")
            completed = subprocess.run(
                ["python3", "-B", str(args.verifier), str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            if completed.returncode:
                rejected += 1
            else:
                print(f"MUTATION_ESCAPED={name}")
    print(f"MUTATION_TESTS={len(MUTATIONS)}")
    print(f"MUTATIONS_REJECTED={rejected}")
    print(f"MUTATIONS_ESCAPED={len(MUTATIONS) - rejected}")
    print(f"MUTATION_GATE_PASS={str(rejected == len(MUTATIONS)).lower()}")
    if rejected != len(MUTATIONS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
