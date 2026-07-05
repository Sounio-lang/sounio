#!/usr/bin/env bash
# Cross-toolchain replication gate for the executable 168-theorem.
#
# WHY: souc v0.80.0 has a documented false-green mode (multi-module compiles -> silent stubs) and
# large-aggregate lowering hazards. Under such a toolchain a bare `PASS` is NOT proof of execution
# (isomorphic to ||ab||<eps not being ab==0). This gate closes that gap: it checks that the 168
# SPECIFIC canonical zero-divisor pairs emitted by souc are set-identical to those computed by an
# INDEPENDENT toolchain (Python, transcribed directly from the Lean spec). No stub reproduces 168
# specific, correct pairs by accident — only counts are forgeable, the pair set is not.
#
# Producers:
#   (1) souc  -> tests/run-pass/sedenion_zd_census_168.sio emits `PAIR ulo uhi uneg vlo vhi vneg`.
#   (2) python-> scripts/research/verify_zd168_oracle.py (from formal/lean4/*.lean cdSigma/primProd).
# Lean's leg is its native_decide-proven counts (prim_count_84 / zd_pair_count_336 /
# zd_projective_count_168) in formal/lean4/SounioZeroDivisorBridge.lean; Lean is not runnable in
# this env, so the element-wise diff is souc-vs-Python (two independent toolchains).
#
# Asserter: /usr/bin/diff (not souc). Exit 0 + CROSS-VERIFIED iff the two 168-sets are identical.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[1/3] souc emits its 168 pairs ..."
./bin/souc run tests/run-pass/sedenion_zd_census_168.sio 2>/dev/null | grep '^PAIR ' | sort -u > "$WORK/souc.txt"
S=$(wc -l < "$WORK/souc.txt")

echo "[2/3] Python oracle (Lean-spec, non-souc) emits its 168 pairs ..."
python3 scripts/research/verify_zd168_oracle.py | grep '^PAIR ' | sort -u > "$WORK/py.txt"
P=$(wc -l < "$WORK/py.txt")

echo "[3/3] independent asserter (diff) checks set equality ..."
if [ "$S" -ne 168 ] || [ "$P" -ne 168 ]; then
  echo "FAIL: expected 168 unique pairs each; souc=$S python=$P"; exit 1
fi
if diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null; then
  echo "CROSS-VERIFIED: souc 168-set == Lean-spec Python-oracle 168-set (168/168 identical pairs)."
  echo "  souc pairs:   $S"
  echo "  python pairs: $P"
  echo "  Lean counts (native_decide-proven): validPrims=84, ordered=336, unordered=168."
  exit 0
else
  echo "MISMATCH: souc and Python disagree on the 168-set. Diff (< souc, > python):"
  diff "$WORK/souc.txt" "$WORK/py.txt" | head -40
  exit 1
fi
