#!/usr/bin/env python3
"""Push the empirical frontier of the CD tower off-seam/ZD converse conjecture to bits=7 (dim 128)
and, if feasible, bits=8 (dim 256).

Conjecture: off_seam(l,u) ==> e_l + e_u is a zero divisor (has a 2-term annihilator).

The oracle's is_zd(l,u) is an O(N^3)-per-pair brute force full 2-term search — ground truth, but
far too slow at bits=7 (N=128 => ~2M (a,b,s) triples per pair, each with an O(N) inner loop =>
~8B ops total) and hopeless at bits=8.

MATHEMATICAL REDUCTION: a 2-term annihilator (e_l+e_u)(e_a + s*e_b) = 0 expands to four output
basis terms at indices l^a, l^b, u^a, u^b, each carrying a sign cd_sigma(*, *). Since a != b and
l != u, the only way these four terms can cancel in pairs is by index coincidence forcing
b = a ^ (l^u) (the unique nontrivial pairing l^a == u^b, l^b == u^a). Given that b, annihilation
holds iff there exists s in {+1,-1} with BOTH:
    cd_sigma(l,a) + s*cd_sigma(u,b) == 0
    s*cd_sigma(l,b) + cd_sigma(u,a) == 0
This gives an O(N)-per-pair test (is_zd_fast) instead of O(N^3) (is_zd_brute).

This script:
  1. Defines cd_sigma (memoized), is_zd_brute (= oracle's is_zd, ground truth), is_zd_fast.
  2. VALIDATES is_zd_fast == is_zd_brute for every lower x upper pair at bits=4,5,6.
     If they ever disagree, STOPS and reports the disagreement (reduction would be broken).
  3. Only if validated, runs the off_seam == is_zd_fast conjecture check at bits=7 (and bits=8
     if it finishes in time), reporting counts and any counterexamples.
"""

import sys
import time
from functools import lru_cache


@lru_cache(maxsize=None)
def cd_sigma(a, b, bits):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH, bH, aL, bL = a >= half, b >= half, a & (half - 1), b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def is_zd_brute(l, u, bits):
    """Ground truth: original O(N^3) full 2-term annihilator search (exactly as in the oracle)."""
    N = 1 << bits
    for a in range(1, N):
        for b in range(a + 1, N):
            for s in (1, -1):
                if all(((cd_sigma(l, a, bits) if (l ^ a) == k else 0)
                        + (s * cd_sigma(l, b, bits) if (l ^ b) == k else 0)
                        + (cd_sigma(u, a, bits) if (u ^ a) == k else 0)
                        + (s * cd_sigma(u, b, bits) if (u ^ b) == k else 0)) == 0
                       for k in range(N)):
                    return True
    return False


def is_zd_fast(l, u, bits):
    """O(N)-per-pair reduction: b is forced to a ^ (l^u); check both sign equations."""
    N = 1 << bits
    xor_lu = l ^ u
    for a in range(1, N):
        b = a ^ xor_lu
        if b == a or b < 1 or b >= N:
            continue
        sla = cd_sigma(l, a, bits)
        slb = cd_sigma(l, b, bits)
        sua = cd_sigma(u, a, bits)
        sub = cd_sigma(u, b, bits)
        for s in (1, -1):
            if sla + s * sub == 0 and s * slb + sua == 0:
                return True
    return False


def off_seam(l, u, bits):
    top = 1 << (bits - 1)
    return not (u == top or (l ^ u) == top)


def validate_fastpath(bits_list):
    for bits in bits_list:
        N = 1 << bits
        top = N // 2
        pairs = [(l, u) for l in range(1, top) for u in range(top, N)]
        t0 = time.time()
        for (l, u) in pairs:
            b_brute = is_zd_brute(l, u, bits)
            b_fast = is_zd_fast(l, u, bits)
            if b_brute != b_fast:
                print(f"FASTPATH MISMATCH at bits={bits} (l,u)=({l},{u}): "
                      f"brute={b_brute} fast={b_fast}")
                return False
        dt = time.time() - t0
        print(f"  bits={bits}: {len(pairs)} pairs checked, brute==fast for all, {dt:.2f}s")
    print("FASTPATH VALIDATED n=4,5,6")
    return True


def run_converse_check(bits, timeout_s=None):
    N = 1 << bits
    top = N // 2
    pairs_count = 0
    off_seam_count = 0
    zd_count = 0
    counterexamples = []
    t0 = time.time()
    for l in range(1, top):
        for u in range(top, N):
            pairs_count += 1
            os_ = off_seam(l, u, bits)
            zd_ = is_zd_fast(l, u, bits)
            if os_:
                off_seam_count += 1
            if zd_:
                zd_count += 1
            if os_ != zd_:
                kind = "off-seam-but-not-ZD" if (os_ and not zd_) else "ZD-but-on-seam"
                counterexamples.append((l, u, kind))
        if timeout_s is not None and time.time() - t0 > timeout_s:
            print(f"  ... TIMEOUT at bits={bits} after {time.time()-t0:.1f}s "
                  f"(l up to {l} of {top}); aborting this bits level.")
            return None
    dt = time.time() - t0
    holds = len(counterexamples) == 0
    print(f"  bits={bits}: N={N}, total pairs={pairs_count}, off-seam={off_seam_count}, "
          f"ZD={zd_count}, equivalence holds={holds}, elapsed={dt:.1f}s")
    if not holds:
        print(f"  FIRST counterexamples (bits={bits}):")
        for (l, u, kind) in counterexamples[:10]:
            print(f"    (l={l}, u={u}): {kind}")
    return {
        "bits": bits,
        "pairs": pairs_count,
        "off_seam": off_seam_count,
        "zd": zd_count,
        "holds": holds,
        "counterexamples": counterexamples,
        "elapsed": dt,
    }


def main():
    print("=== Step 1: validate is_zd_fast against is_zd_brute (ground truth) at bits=4,5,6 ===")
    ok = validate_fastpath([4, 5, 6])
    if not ok:
        print("ABORT: fast-path reduction disagrees with brute force. Do not trust bits=7/8 results.")
        sys.exit(1)

    print()
    print("=== Step 2: converse conjecture check at bits=7 (dim 128) using is_zd_fast ===")
    result7 = run_converse_check(7, timeout_s=540)

    result8 = None
    if result7 is not None:
        print()
        print("=== Step 3: converse conjecture check at bits=8 (dim 256) using is_zd_fast ===")
        result8 = run_converse_check(8, timeout_s=540)
    else:
        print("Skipping bits=8: bits=7 did not complete.")

    print()
    print("=== FINAL VERDICT ===")
    if result7 is None:
        print("CONVERSE n=7: SKIPPED (timeout)")
    elif result7["holds"]:
        print("CONVERSE n=7: HOLDS "
              f"(pairs={result7['pairs']}, off_seam={result7['off_seam']}, zd={result7['zd']})")
    else:
        cex = "; ".join(f"(l={l},u={u},{kind})" for (l, u, kind) in result7["counterexamples"][:5])
        print(f"CONVERSE n=7: REFUTED (counterexamples: {cex})")

    if result8 is None:
        print("CONVERSE n=8: SKIPPED (timeout or bits=7 incomplete)")
    elif result8["holds"]:
        print("CONVERSE n=8: HOLDS "
              f"(pairs={result8['pairs']}, off_seam={result8['off_seam']}, zd={result8['zd']})")
    else:
        cex = "; ".join(f"(l={l},u={u},{kind})" for (l, u, kind) in result8["counterexamples"][:5])
        print(f"CONVERSE n=8: REFUTED (counterexamples: {cex})")


if __name__ == "__main__":
    main()
