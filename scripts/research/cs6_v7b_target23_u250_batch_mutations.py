#!/usr/bin/env python3
"""Check that the U250 receipt verifier rejects critical tampering."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from cs6_v7b_target23_u250_batch_verify import verify


MUTATIONS = {
    "bit_mismatch": ("BIT_MISMATCHES=0", "BIT_MISMATCHES=1"),
    "leaf_shortfall": ("LEAVES=331", "LEAVES=330"),
    "fake_fpga": ("FPGA_EXECUTION=true", "FPGA_EXECUTION=false"),
    "baseline_replay": ("EXECUTED_XCLBIN_UUID=", "EXECUTED_XCLBIN_UUID=13259b30-d0d2-d4db-deba-bfc0153a26d2#"),
    "arb_shortfall": ("ARB_CENTER_SIGN_CERTIFICATES=3", "ARB_CENTER_SIGN_CERTIFICATES=2"),
    "claim_escalation": ("GLOBAL_HPG_CERTIFICATE=false", "GLOBAL_HPG_CERTIFICATE=true"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    rejected = 0
    for mutation, (old, new) in MUTATIONS.items():
        with tempfile.TemporaryDirectory(prefix=f"target23-u250-{mutation}-") as temporary:
            candidate = Path(temporary) / "receipt"
            shutil.copytree(args.receipt, candidate)
            summary = candidate / "summary.txt"
            text = summary.read_text(encoding="ascii")
            if old not in text:
                raise SystemExit(f"mutation anchor missing: {mutation}")
            if old.endswith("="):
                lines = text.splitlines()
                matches = [index for index, line in enumerate(lines) if line.startswith(old)]
                if len(matches) != 1:
                    raise SystemExit(f"mutation line anchor mismatch: {mutation}")
                lines[matches[0]] = new.removesuffix("#")
                mutated = "\n".join(lines) + "\n"
            else:
                mutated = text.replace(old, new, 1)
            summary.write_text(mutated, encoding="ascii")
            try:
                verify(candidate)
            except (OSError, KeyError, ValueError):
                rejected += 1
                print(f"MUTATION={mutation}\tREJECTED=true")
            else:
                print(f"MUTATION={mutation}\tREJECTED=false")
    print("SCHEMA=sounio.cs6.v7b-target23-u250-batch-mutations.v1")
    print(f"MUTATIONS_TOTAL={len(MUTATIONS)}")
    print(f"MUTATIONS_REJECTED={rejected}")
    print(f"TARGET23_U250_BATCH_MUTATIONS_PASS={str(rejected == len(MUTATIONS)).lower()}")
    if rejected != len(MUTATIONS):
        raise SystemExit("U250 receipt mutation escaped")


if __name__ == "__main__":
    main()
