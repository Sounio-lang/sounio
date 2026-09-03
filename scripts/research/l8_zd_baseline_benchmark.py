#!/usr/bin/env python3
"""Baseline benchmark: existing Python implementation at level 8 (256-dim).

Times each phase of the current routon_zd_contract machinery applied to b=8:
  1. sign-table build (pure Python cds loop, 256x256)
  2. exact 2-cycle census scan (exact_nullity_index_pairs)
  3. SVD reference verification (svd_zd_index_pairs) — the audited exactness
     oracle of the L4-L7 contracts
This is the implementation to beat.  Prints machine-readable timings.
"""

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routon_zd_contract as R


def main():
    b = 8
    t0 = time.perf_counter()
    S = R.get_sign_matrix(b)
    t1 = time.perf_counter()
    print(f"BASELINE_SIGN_TABLE seconds={t1 - t0:.3f}")

    t2 = time.perf_counter()
    exact = R.exact_nullity_index_pairs(b)
    t3 = time.perf_counter()
    n_pairs = len(exact)
    hist = dict(sorted(Counter(exact.values()).items()))
    print(f"BASELINE_EXACT_SCAN seconds={t3 - t2:.3f} index_pairs={n_pairs}")
    print(f"BASELINE_EXACT_SCAN triples={2 * n_pairs} law_Z8={R.census_law(8)}")
    print(f"BASELINE_HISTOGRAM {hist}")

    t4 = time.perf_counter()
    svd = R.svd_zd_index_pairs(b)
    t5 = time.perf_counter()
    ok = set(exact.keys()) == svd
    print(f"BASELINE_SVD_VERIFY seconds={t5 - t4:.3f} svd_pairs={len(svd)} "
          f"census_equal={ok}")
    print(f"BASELINE_TOTAL seconds={t5 - t0:.3f}")
    return 0 if (2 * n_pairs == R.census_law(8) and ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
