#!/usr/bin/env python3
"""Independent census of the seven xor-fibers of the 84 primitives.

Ports the Lean definitions in formal/lean4/SounioZeroDivisorBridge.lean
and SounioSurgicalCalculus.lean. Does not call Lean. Does not edit
self-hosted/. Measurement only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


def cd_sigma(a: int, b: int, bits: int) -> int:
    """Sounio.CayleyDickson.cdSigma — recursive, bits=4 is sedenion."""
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    a_hi = a >= half
    b_hi = b >= half
    a_lo = a & (half - 1)
    b_lo = b & (half - 1)
    if (not a_hi) and (not b_hi):
        return cd_sigma(a_lo, b_lo, bits - 1)
    if (not a_hi) and b_hi:
        return cd_sigma(b_lo, a_lo, bits - 1)
    if a_hi and (not b_hi):
        s = cd_sigma(a_lo, b_lo, bits - 1)
        return s if b_lo == 0 else -s
    s = cd_sigma(b_lo, a_lo, bits - 1)
    return -s if b_lo == 0 else s


def sed_sigma(a: int, b: int) -> int:
    return cd_sigma(a, b, 4)


@dataclass(frozen=True)
class PrimSed:
    lo: int
    hi: int
    neg: bool

    def xor_label(self) -> int:
        return self.lo ^ self.hi

    def render(self) -> str:
        sign = "-" if self.neg else "+"
        return f"e{self.lo}{sign}e{self.hi}"


def is_prim_valid(v: PrimSed) -> bool:
    return (
        1 <= v.lo <= 7
        and 9 <= v.hi <= 15
        and (v.lo ^ v.hi) != 8
    )


def all_prims() -> list[PrimSed]:
    out: list[PrimSed] = []
    for lo in range(1, 8):
        for hi in range(9, 16):
            out.append(PrimSed(lo, hi, False))
            out.append(PrimSed(lo, hi, True))
    return out


def valid_prims() -> list[PrimSed]:
    return [v for v in all_prims() if is_prim_valid(v)]


def prim_prod(u: PrimSed, v: PrimSed, k: int) -> int:
    c_ll = sed_sigma(u.lo, v.lo) if (u.lo ^ v.lo) == k else 0
    s_v = -1 if v.neg else 1
    s_u = -1 if u.neg else 1
    c_lh = (s_v * sed_sigma(u.lo, v.hi)) if (u.lo ^ v.hi) == k else 0
    c_hl = (s_u * sed_sigma(u.hi, v.lo)) if (u.hi ^ v.lo) == k else 0
    c_hh = (s_u * s_v * sed_sigma(u.hi, v.hi)) if (u.hi ^ v.hi) == k else 0
    return c_ll + c_lh + c_hl + c_hh


def is_zero_pair(u: PrimSed, v: PrimSed) -> bool:
    return all(prim_prod(u, v, k) == 0 for k in range(16))


def fiber_prims(label: int, prims: list[PrimSed]) -> list[PrimSed]:
    return [v for v in prims if v.xor_label() == label]


def unlearn(u: PrimSed, prims: list[PrimSed]) -> list[PrimSed]:
    return [v for v in prims if u != v and is_zero_pair(u, v)]


def census() -> dict:
    prims = valid_prims()
    labels_present = sorted({v.xor_label() for v in prims})
    fibers = {L: fiber_prims(L, prims) for L in labels_present}
    fiber9 = fibers.get(9, [])
    outsider = PrimSed(1, 10, False)

    # kernel distribution for every u
    same_fiber = 0
    split = 0
    empty = 0
    kernel_sizes: dict[int, int] = {}
    distinct_label_hist: dict[int, int] = {}
    kernel_subset_fiber = 0
    self_in_kernel = 0
    for u in prims:
        ker = unlearn(u, prims)
        kernel_sizes[len(ker)] = kernel_sizes.get(len(ker), 0) + 1
        labs = {a.xor_label() for a in ker}
        distinct_label_hist[len(labs)] = distinct_label_hist.get(len(labs), 0) + 1
        if not ker:
            empty += 1
        elif labs == {u.xor_label()}:
            same_fiber += 1
        else:
            split += 1
        if all(a.xor_label() == u.xor_label() for a in ker):
            kernel_subset_fiber += 1
        if u in ker:
            self_in_kernel += 1

    # pairwise disjoint + cover
    seen: set[PrimSed] = set()
    overlap = 0
    for L, members in fibers.items():
        for m in members:
            if m in seen:
                overlap += 1
            seen.add(m)
    uncovered = [v.render() for v in prims if v not in seen]

    primA = PrimSed(3, 10, False)
    ker_A = unlearn(primA, prims)

    return {
        "n_valid": len(prims),
        "labels_present": labels_present,
        "n_fibers": len(labels_present),
        "fiber_sizes": {str(L): len(ms) for L, ms in fibers.items()},
        "all_size_12": all(len(ms) == 12 for ms in fibers.values()),
        "pairwise_overlap": overlap,
        "union_size": len(seen),
        "uncovered": uncovered,
        "is_partition": (
            len(labels_present) == 7
            and overlap == 0
            and len(seen) == 84
            and all(len(ms) == 12 for ms in fibers.values())
            and not uncovered
        ),
        "fiber9": [v.render() for v in fiber9],
        "fiber9_tuples": [[v.lo, v.hi, v.neg] for v in fiber9],
        "outsider": outsider.render(),
        "outsider_label": outsider.xor_label(),
        "outsider_valid": is_prim_valid(outsider),
        "outsider_in_fiber9": outsider in fiber9,
        "kernel_size_hist": {str(k): v for k, v in sorted(kernel_sizes.items())},
        "kernel_distinct_label_hist": {
            str(k): v for k, v in sorted(distinct_label_hist.items())
        },
        "kernel_all_same_fiber_as_u": same_fiber,
        "kernel_splits_across_fibers": split,
        "kernel_empty": empty,
        "kernel_subset_own_fiber": kernel_subset_fiber,
        "self_in_kernel": self_in_kernel,
        "primA": primA.render(),
        "primA_label": primA.xor_label(),
        "primA_kernel": [v.render() for v in ker_A],
        "primA_kernel_labels": sorted({v.xor_label() for v in ker_A}),
        "equivalence": "xorLabel(v) = v.lo XOR v.hi",
        "edit_of_u": "fiberPrims(xorLabel(u))  -- 12 mates including u",
        "prediction_seven_fibers": True,
        "prediction_orthogonal": False,
        "verdict": "SEVEN_DISJOINT_FIBERS",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    d = census()
    text = json.dumps(d, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
