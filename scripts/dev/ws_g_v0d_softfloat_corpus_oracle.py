#!/usr/bin/env python3
"""V0-D hard-case corpus oracle + widen-f64 trap classification.

Corpus: tests/vectors/f128_f256_v0d/arith_hard_f{128,256}.jsonl (MPFR RNE).
Does not implement Sounio softfloat. Exit 0 only when:
  - corpus hashes/counts OK
  - trap sets are non-empty and internally consistent
  - a softfloat corpus consumer is present

Prints PASS/FAIL/NOTE and two explicit lists:
  CATCH_WIDEN_F64 / MISS_WIDEN_F64
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VEC = ROOT / "tests" / "vectors" / "f128_f256_v0d"
RUNNER = ROOT / "scripts" / "dev" / "ws_g_v0d_softfloat_corpus_runner.py"

EXPECTED_MD5 = {
    # 2026-09-06: sticky sub rows f128_arith_0010 / f256_arith_0063 result.limbs
    # corrected to IEEE ops on encoded wire operands (generator previously used
    # unrepresentable exact MPFR inputs; encode(a)≠a made result unreachable).
    "arith_hard_f128.jsonl": "4e49d07307dceff293fa2fc0666486a7",
    "arith_hard_f256.jsonl": "bef3744e774292ab622448bc2d6b63f1",
}

EXPECTED_COUNTS = {
    "arith_hard_f128.jsonl": 53,
    "arith_hard_f256.jsonl": 50,
}

# Families that grade IEEE structure a widen-f64 (or short-precision) path fails.
# Easy exact cases (sqrt(4)=2, some overflow-to-inf) can still match after f64.
STRUCTURAL_TRAP_FAMILIES = frozenset(
    {
        "halfway_tie_even",
        "sticky_bit",
        "catastrophic_cancel",
        "rump",
    }
)

# Families that often still pass under a widen-f64 shortcut on this corpus.
# Still required for bit-identity green, but alone they do not prove softfloat.
WEAK_VS_WIDEN_F64_FAMILIES = frozenset(
    {
        "sqrt_hard",
        "overflow_underflow",
    }
)

CONSUMER_MARKERS = [
    ROOT / "self-hosted/compiler/f128_f256_v0d_softfloat_corpus_probe.sio",
    ROOT / "tests/run-pass/f128_v0d_softfloat_corpus_smoke.sio",
    ROOT / "scripts/dev/ws_g_v0d_softfloat_corpus_runner.py",
]


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_rows(name: str) -> list[dict]:
    path = VEC / name
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def limbs_equal(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return False
    return list(a.get("limbs") or []) == list(b.get("limbs") or [])


def classify_widen_f64_trap(r: dict) -> tuple[bool, list[str]]:
    """Return (would_catch_widen_f64, reasons).

    A widen-f64 softfloat does: decode→f64, host op, re-encode to binaryN.
    It is caught when the MPFR result limbs differ from that path's limbs, or
    when the family is structurally sensitive beyond binary64.
    """
    reasons: list[str] = []
    fam = r.get("family") or ""
    if fam in STRUCTURAL_TRAP_FAMILIES:
        reasons.append(f"family={fam}")
    if r.get("f64_bits_differ") is True:
        reasons.append("f64_bits_differ")
    if r.get("f64_sign_differs") is True:
        reasons.append("f64_sign_differs")
    # Rump rows carry explicit f64_result from the generator.
    if "f64_result" in r and not limbs_equal(r.get("result"), r.get("f64_result")):
        reasons.append("f64_result_limbs_ne_result")
    if fam == "rump" or "rump" in (r.get("op") or ""):
        if "family=rump" not in reasons:
            reasons.append("family=rump")
    # Deduce catch: any reason except we still list weak families as non-catch
    # unless f64_bits_differ / f64_result proves otherwise.
    if reasons:
        return True, reasons
    if fam in WEAK_VS_WIDEN_F64_FAMILIES:
        return False, [f"family={fam}_may_match_f64_path"]
    return False, ["unclassified_assume_weak"]


def main() -> int:
    rc = 0
    all_catch: list[tuple[str, str, str, list[str]]] = []
    all_miss: list[tuple[str, str, str, list[str]]] = []
    must_trap_ids: list[str] = []

    if not VEC.is_dir():
        print(f"FAIL missing_corpus_dir {VEC}")
        return 1

    for fname, n_exp in EXPECTED_COUNTS.items():
        path = VEC / fname
        if not path.is_file():
            print(f"FAIL missing_corpus {fname}")
            rc = 1
            continue
        got = md5_file(path)
        exp = EXPECTED_MD5[fname]
        if got != exp:
            print(f"FAIL corpus_md5_mismatch {fname} got={got} expected={exp}")
            rc = 1
        else:
            print(f"PASS corpus_md5_ok {fname}")

        rows = load_rows(fname)
        if len(rows) != n_exp:
            print(f"FAIL corpus_count {fname} n={len(rows)} expected={n_exp}")
            rc = 1
        else:
            print(f"PASS corpus_loaded {fname} n={len(rows)}")

        fam_counts: dict[str, int] = {}
        for r in rows:
            fam = r.get("family") or "?"
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            if r.get("rounding") != "rne":
                print(f"FAIL non_rne_rounding {r.get('id')}")
                rc = 1
            if "result" not in r or "limbs" not in r["result"]:
                print(f"FAIL missing_result_limbs {r.get('id')}")
                rc = 1
            if "provenance" not in r or r["provenance"].get("tool") != "MPFR":
                print(f"FAIL missing_mpfr_provenance {r.get('id')}")
                rc = 1

            catches, reasons = classify_widen_f64_trap(r)
            entry = (r["id"], r.get("op", "?"), fam, reasons)
            if catches:
                all_catch.append(entry)
                must_trap_ids.append(r["id"])
            else:
                all_miss.append(entry)

        print(
            f"PASS family_coverage {fname} "
            + " ".join(f"{k}={v}" for k, v in sorted(fam_counts.items()))
        )

        # Hard requirements: trap families must be present
        for need in ("halfway_tie_even", "sticky_bit", "catastrophic_cancel", "rump"):
            if fam_counts.get(need, 0) < 1:
                print(f"FAIL missing_required_trap_family {fname} {need}")
                rc = 1
            else:
                print(f"PASS trap_family_present {fname} {need}={fam_counts[need]}")

        # Rump must disagree with f64_result (correctness vs plausible)
        rump_rows = [r for r in rows if r.get("family") == "rump"]
        for r in rump_rows:
            if "f64_result" not in r:
                print(f"FAIL rump_missing_f64_result {r['id']}")
                rc = 1
            elif limbs_equal(r["result"], r["f64_result"]):
                print(f"FAIL rump_f64_result_equals_exact {r['id']}")
                rc = 1
            else:
                print(f"PASS rump_exact_ne_f64_result {r['id']}")
            if not r.get("f64_bits_differ", False) and not (
                "f64_result" in r and not limbs_equal(r["result"], r["f64_result"])
            ):
                print(f"FAIL rump_not_marked_f64_bits_differ {r['id']}")
                rc = 1

    print(f"PASS widen_f64_trap_count catch={len(all_catch)} miss={len(all_miss)}")
    if len(all_catch) < 1:
        print("FAIL empty_widen_f64_trap_set")
        rc = 1

    print("CATCH_WIDEN_F64_BEGIN")
    for vid, op, fam, reasons in all_catch:
        print(f"CATCH_WIDEN_F64 {vid} op={op} family={fam} reasons={','.join(reasons)}")
    print("CATCH_WIDEN_F64_END")

    print("MISS_WIDEN_F64_BEGIN")
    for vid, op, fam, reasons in all_miss:
        print(f"MISS_WIDEN_F64 {vid} op={op} family={fam} reasons={','.join(reasons)}")
    print("MISS_WIDEN_F64_END")

    # Must-trap ID list for future consumer: every CATCH id is mandatory.
    print(f"PASS must_trap_ids n={len(must_trap_ids)}")
    print("MUST_TRAP_IDS " + ",".join(must_trap_ids))

    # Correctness assertion shape (what a green consumer must print):
    print(
        "PASS correctness_contract "
        "bit_identity_on_all_rows_and_must_trap_ids; "
        "rump_uses_result_not_f64_result; "
        "halfway_sticky_cancel_required"
    )

    found = [p for p in CONSUMER_MARKERS if p.is_file()]
    if not found:
        print(
            "FAIL v0d_softfloat_does_not_consume_hard_corpus "
            "no consumer at "
            + ",".join(str(p.relative_to(ROOT)) for p in CONSUMER_MARKERS)
            + f" (arith_hard_f128.jsonl n=53 + arith_hard_f256.jsonl n=50 unconsumed; "
            f"must_trap_ids n={len(must_trap_ids)} untested)"
        )
        rc = 1
    else:
        print(
            "PASS v0d_softfloat_consumer_present "
            + ",".join(str(p.relative_to(ROOT)) for p in found)
        )

    # Hard bar: runner must execute and assert bit-identity (file presence alone
    # is not green — V0-D “de verdade”).
    if not RUNNER.is_file():
        print("FAIL v0d_softfloat_runner_missing scripts/dev/ws_g_v0d_softfloat_corpus_runner.py")
        rc = 1
    else:
        proc = subprocess.run(
            [sys.executable, str(RUNNER)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        for line in (proc.stdout or "").splitlines():
            print(line)
        if proc.stderr:
            for line in proc.stderr.splitlines():
                print(f"NOTE runner_stderr {line}")
        if proc.returncode != 0:
            print(
                "FAIL v0d_softfloat_bit_identity_runner "
                f"exit={proc.returncode} (limb softfloat must match MPFR result.limbs "
                f"on all hard rows including must_trap_ids n={len(must_trap_ids)})"
            )
            rc = 1
        elif "bit_identity=all_hard_rows" not in (proc.stdout or ""):
            print("FAIL v0d_softfloat_runner_missing_bit_identity_receipt")
            rc = 1
        else:
            print("PASS v0d_softfloat_bit_identity_runner")

    return rc


if __name__ == "__main__":
    sys.exit(main())
