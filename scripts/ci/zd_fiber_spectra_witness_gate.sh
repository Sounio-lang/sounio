#!/usr/bin/env bash
# scripts/ci/zd_fiber_spectra_witness_gate.sh
#
# A PRODUCTION gate that emits both a verdict token and a WITNESS.
#
# The proposition: the number of distinct adjacency spectra over the ZD fibers
# of the Cayley-Dickson tower is 3*2^(n-5), for n = 5, 6, 7.
#
# The witness: a sha256 over those spectra themselves, sorted. Without it the
# token is blind to any change that keeps the COUNT while replacing the spectra
# -- which is not hypothetical here. R15 measured exactly such a change on this
# object (a single sign flip, sigma(H/2, H+H/2)) and R16 showed it preserves the
# whole classification while relabelling every block. The token held; the
# geometries were all different.
#
# Emits:
#   ZD_FIBER_SPECTRA_VERDICT <token>
#   ZD_FIBER_SPECTRA_WITNESS <sha256>
#
# Budget: ~3 s, well inside the executor's 30 s per-gate cap.
# Exit 0 = the proposition holds; non-zero = it does not.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# SFC_ZD_FIBER_PERTURB, when set to "1", applies the count-preserving sign flip
# R15/R16 characterised. It exists so the witness mechanism can be DEMONSTRATED
# rather than asserted: with it set, this gate still exits 0 and still emits the
# same verdict token, and only the witness moves.
exec python3 - "${SFC_ZD_FIBER_PERTURB:-0}" <<'PY'
import hashlib
import importlib.util
import sys
from pathlib import Path

REPO = Path.cwd()
spec = importlib.util.spec_from_file_location(
    "r15", REPO / "scripts/research/self_falsifying_compilation_line_r15_contract.py")
r15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r15)

perturb = sys.argv[1] == "1"

ok = True
digest = hashlib.sha256()
for n in (5, 6, 7):
    H = 1 << (n - 1)
    flip = (H // 2, H + H // 2) if perturb else None
    S = sorted(r15.spectra(n, flip))
    want = 3 * 2 ** (n - 5)
    if len(S) != want:
        ok = False
        print(f"n={n}: {len(S)} distinct spectra, expected {want}")
    else:
        print(f"n={n}: {len(S)} distinct spectra = 3*2^(n-5)")
    # the witness is the LABELLING, not its size
    for s in S:
        digest.update(repr(s).encode())
        digest.update(b"|")

print()
print(f"ZD_FIBER_SPECTRA_VERDICT "
      f"{'SPECTRA_COUNT_IS_3_TIMES_2_POW_N_MINUS_5' if ok else 'COUNT_LAW_FAILS'}")
print(f"ZD_FIBER_SPECTRA_WITNESS {digest.hexdigest()}")
sys.exit(0 if ok else 1)
PY
