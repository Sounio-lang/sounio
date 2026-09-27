#!/usr/bin/env python3
"""V0-D softfloat corpus runner — bit-identity vs MPFR hard rows.

Consumes tests/vectors/f128_f256_v0d/arith_hard_f{128,256}.jsonl through
compiler-owned limb softfloat (scripts/dev/softfloat_limb.py). Exit 0 only
when every row matches result.limbs (RNE), including MUST_TRAP families.

Does not use MPFR, host libm, or a widen-f64 shortcut on the payload path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dev"))

from softfloat_limb import FMT_BY_NAME, apply_op  # noqa: E402

VEC = ROOT / "tests" / "vectors" / "f128_f256_v0d"
CORPORA = ("arith_hard_f128.jsonl", "arith_hard_f256.jsonl")

STRUCTURAL_TRAP_FAMILIES = frozenset(
    {"halfway_tie_even", "sticky_bit", "catastrophic_cancel", "rump"}
)


def load_rows(name: str) -> list[dict]:
    path = VEC / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    total = 0
    ok = 0
    fail = 0
    trap_ok = 0
    trap_fail = 0
    failures: list[str] = []

    for fname in CORPORA:
        rows = load_rows(fname)
        for r in rows:
            total += 1
            fmt = FMT_BY_NAME[r["format"]]
            op = r["op"]
            a = list(r["a"]["limbs"])
            b = list(r["b"]["limbs"]) if r.get("b") is not None and "limbs" in r.get("b", {}) else None
            if r.get("arity", 2) == 1 or op.endswith("_sqrt"):
                b = None
            got = apply_op(op, a, b, fmt)
            exp = list(r["result"]["limbs"])
            fam = r.get("family") or ""
            is_trap = fam in STRUCTURAL_TRAP_FAMILIES
            if got == exp:
                ok += 1
                if is_trap:
                    trap_ok += 1
            else:
                fail += 1
                if is_trap:
                    trap_fail += 1
                msg = (
                    f"FAIL bit_identity {r['id']} op={op} family={fam} "
                    f"got={got} expected={exp}"
                )
                failures.append(msg)
                print(msg)

    print(f"PASS softfloat_runner_rows total={total} ok={ok} fail={fail}")
    print(f"PASS softfloat_runner_must_trap trap_ok={trap_ok} trap_fail={trap_fail}")

    if fail == 0 and total > 0:
        print(
            f"PASS f128_f256_v0d_softfloat_corpus_runner "
            f"bit_identity=all_hard_rows n={total} "
            f"must_trap_ok={trap_ok} engine=limb_bigint_rne"
        )
        return 0

    print(
        f"FAIL f128_f256_v0d_softfloat_corpus_runner "
        f"bit_identity_misses={fail} total={total} trap_fail={trap_fail}"
    )
    for line in failures[:20]:
        print(line)
    if len(failures) > 20:
        print(f"NOTE truncated_failures showing=20 of={len(failures)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
