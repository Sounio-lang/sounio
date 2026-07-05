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

# The Python oracle emits BOTH faces (PAIR = ZD, TRIPLE = non-Fano) in one run.
python3 scripts/research/verify_zd168_oracle.py > "$WORK/py_all.txt"

echo "[face 1/2] zero-divisor census: souc vs oracle ..."
./bin/souc run tests/run-pass/sedenion_zd_census_168.sio 2>/dev/null | grep '^PAIR ' | sort -u > "$WORK/souc_zd.txt"
grep '^PAIR ' "$WORK/py_all.txt" | sort -u > "$WORK/py_zd.txt"
SZD=$(wc -l < "$WORK/souc_zd.txt"); PZD=$(wc -l < "$WORK/py_zd.txt")

echo "[face 2/2] non-Fano census: souc vs oracle ..."
./bin/souc run tests/run-pass/octonion_nonfano_census_168.sio 2>/dev/null | grep '^TRIPLE ' | sort -u > "$WORK/souc_nf.txt"
grep '^TRIPLE ' "$WORK/py_all.txt" | sort -u > "$WORK/py_nf.txt"
SNF=$(wc -l < "$WORK/souc_nf.txt"); PNF=$(wc -l < "$WORK/py_nf.txt")

fail=0
if [ "$SZD" -ne 168 ] || [ "$PZD" -ne 168 ] || ! diff -q "$WORK/souc_zd.txt" "$WORK/py_zd.txt" >/dev/null; then
  echo "MISMATCH (ZD): souc=$SZD python=$PZD"; diff "$WORK/souc_zd.txt" "$WORK/py_zd.txt" | head -20; fail=1
fi
if [ "$SNF" -ne 168 ] || [ "$PNF" -ne 168 ] || ! diff -q "$WORK/souc_nf.txt" "$WORK/py_nf.txt" >/dev/null; then
  echo "MISMATCH (non-Fano): souc=$SNF python=$PNF"; diff "$WORK/souc_nf.txt" "$WORK/py_nf.txt" | head -20; fail=1
fi
if [ "$fail" -eq 0 ]; then
  echo "CROSS-VERIFIED: BOTH faces of the 168-theorem, element-wise, souc == Lean-spec Python oracle."
  echo "  zero-divisor classes: $SZD/168 identical pairs"
  echo "  non-Fano triples:     $SNF/168 identical triples"
  echo "  Lean (native_decide-proven): validPrims=84, ordered=336, unordered=168, nonFanoCount=168,"
  echo "  and nonfano_zd_bridge proves the two 168s are equal (= |PSL(2,7)|)."
  exit 0
fi
exit 1
