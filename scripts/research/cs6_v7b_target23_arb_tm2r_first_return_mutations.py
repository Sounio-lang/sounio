#!/usr/bin/env python3
"""Adversarial receipt mutations for the Arb TM2R first return."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


MUTATIONS = (
    ("source_hash", "WORKER_SOURCE_SHA256=", "WORKER_SOURCE_SHA256=" + "0" * 64),
    ("leaf_id", "LEAF_ID=U08-0000000223_S09-0000000325", "LEAF_ID=U00-0000000000_S00-0000000000"),
    ("source_degree", "SOURCE_DEGREE=2", "SOURCE_DEGREE=1"),
    ("source_variables", "SOURCE_VARIABLES=2", "SOURCE_VARIABLES=1"),
    ("residual_variables", "RESIDUAL_VARIABLES=4", "RESIDUAL_VARIABLES=3"),
    ("reconditioning", "RECONDITIONING=QR_DERIVED_RATIONAL_BASIS_ZONOTOPE_HULL_EVERY_STEP", "RECONDITIONING=NONE"),
    ("completed_steps", "COMPLETED_STEPS=617", "COMPLETED_STEPS=616"),
    ("picard", "PICARD_CONTAINMENTS=617", "PICARD_CONTAINMENTS=616"),
    ("endpoints", "ENDPOINT_PICARD_CONTAINMENTS=617", "ENDPOINT_PICARD_CONTAINMENTS=616"),
    ("reconditionings", "RECONDITIONINGS=617", "RECONDITIONINGS=616"),
    ("generators", "GENERATOR_RECONSTRUCTIONS=15810", "GENERATOR_RECONSTRUCTIONS=0"),
    ("events", "EVENTS_VALIDATED=1", "EVENTS_VALIDATED=0"),
    ("initial_departure", "INITIAL_DEPARTURE_TUBES=1", "INITIAL_DEPARTURE_TUBES=0"),
    ("prior_downward", "PRIOR_DOWNWARD_TUBES=1", "PRIOR_DOWNWARD_TUBES=0"),
    ("zero_free", "ZERO_FREE_PRIOR_TUBES=614", "ZERO_FREE_PRIOR_TUBES=613"),
    ("failure", "FAILURE_CLASS=NONE", "FAILURE_CLASS=PICARD_NO_CLOSURE"),
    ("event_step", "FIRST_RETURN_END_STEP=617", "FIRST_RETURN_END_STEP=616"),
    ("event_lower", "FIRST_RETURN_TIME_LOWER_Q=77/32", "FIRST_RETURN_TIME_LOWER_Q=615/256"),
    ("before_sign", "FIRST_RETURN_W_BEFORE_UPPER_Q=", "FIRST_RETURN_W_BEFORE_UPPER_Q=0"),
    ("after_sign", "FIRST_RETURN_W_AFTER_LOWER_Q=", "FIRST_RETURN_W_AFTER_LOWER_Q=0"),
    ("normal_sign", "FIRST_RETURN_NORMAL_LOWER_Q=", "FIRST_RETURN_NORMAL_LOWER_Q=0"),
    ("derivative_sign", "FIRST_RETURN_W_DERIVATIVE_LOWER_Q=", "FIRST_RETURN_W_DERIVATIVE_LOWER_Q=0"),
    ("first_return", "FULL_LEAF_FIRST_RETURN_CERTIFICATE=true", "FULL_LEAF_FIRST_RETURN_CERTIFICATE=false"),
    ("second_return", "FULL_LEAF_SECOND_RETURN_CERTIFICATE=false", "FULL_LEAF_SECOND_RETURN_CERTIFICATE=true"),
    ("point_fallback", "POINT_FALLBACK_USED=false", "POINT_FALLBACK_USED=true"),
    ("chaos", "CHAOS_PROVED=false", "CHAOS_PROVED=true"),
    ("attractor", "CHAOTIC_ATTRACTOR_PROVED=false", "CHAOTIC_ATTRACTOR_PROVED=true"),
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
    with tempfile.TemporaryDirectory(prefix="cs6-arb-tm2r-first-return-mutations-") as raw:
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
