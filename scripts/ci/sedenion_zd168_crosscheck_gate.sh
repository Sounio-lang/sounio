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

# The Python oracle emits all five faces (PAIR=ZD, TRIPLE=non-Fano, ARROW=dagger, MEASURE=E/Var, SWEEP=i64 boundary).
python3 scripts/research/verify_zd168_oracle.py > "$WORK/py_all.txt"

echo "[face 1/5] zero-divisor census: souc vs oracle ..."
./bin/souc run tests/run-pass/sedenion_zd_census_168.sio 2>/dev/null | grep '^PAIR ' | sort -u > "$WORK/souc_zd.txt"
grep '^PAIR ' "$WORK/py_all.txt" | sort -u > "$WORK/py_zd.txt"
SZD=$(wc -l < "$WORK/souc_zd.txt"); PZD=$(wc -l < "$WORK/py_zd.txt")

echo "[face 2/5] non-Fano census: souc vs oracle ..."
./bin/souc run tests/run-pass/octonion_nonfano_census_168.sio 2>/dev/null | grep '^TRIPLE ' | sort -u > "$WORK/souc_nf.txt"
grep '^TRIPLE ' "$WORK/py_all.txt" | sort -u > "$WORK/py_nf.txt"
SNF=$(wc -l < "$WORK/souc_nf.txt"); PNF=$(wc -l < "$WORK/py_nf.txt")

echo "[face 3/5] 84<->84 dagger bijection map: souc vs oracle ..."
./bin/souc run tests/run-pass/octonion_dagger_bijection_84.sio 2>/dev/null | grep '^ARROW ' | sort -u > "$WORK/souc_ar.txt"
grep '^ARROW ' "$WORK/py_all.txt" | sort -u > "$WORK/py_ar.txt"
SAR=$(wc -l < "$WORK/souc_ar.txt"); PAR=$(wc -l < "$WORK/py_ar.txt")

echo "[face 4/5] measure-layer exact E/Var over Q: souc vs Python fractions ..."
./bin/souc run tests/run-pass/sedenion_measure_annihilation_exact.sio 2>/dev/null | grep '^MEASURE ' | sort > "$WORK/souc_me.txt"
grep '^MEASURE ' "$WORK/py_all.txt" | sort > "$WORK/py_me.txt"
SME=$(wc -l < "$WORK/souc_me.txt"); PME=$(wc -l < "$WORK/py_me.txt")

echo "[face 5/5] generalized sweep + i64 boundary: souc (overflow-censored) vs unbounded oracle ..."
./bin/souc run tests/run-pass/sedenion_measure_annihilation_general.sio 2>/dev/null | grep '^SCALE ' > "$WORK/souc_sw.txt"
grep '^SWEEP ' "$WORK/py_all.txt" > "$WORK/py_sw.txt"
sweep_fail=0
boundary=0
for k in $(seq 1 12); do
  s=$(grep "^SCALE $k " "$WORK/souc_sw.txt" || true)
  p=$(grep "^SWEEP $k " "$WORK/py_sw.txt" || true)
  pn=$(echo "$p" | awk '{print $3}'); pd=$(echo "$p" | awk '{print $4}'); pf=$(echo "$p" | awk '{print $5}')
  if echo "$s" | grep -q OVERFLOW; then
    [ "$pf" = "BIGINT" ] || { echo "MISMATCH sweep k=$k: souc OVERFLOW but oracle FITS i64 ($pn/$pd)"; sweep_fail=1; }
  else
    sn=$(echo "$s" | awk '{print $4}'); sd=$(echo "$s" | awk '{print $5}')
    { [ "$sn" = "$pn" ] && [ "$sd" = "$pd" ]; } || { echo "MISMATCH sweep k=$k: souc $sn/$sd vs oracle $pn/$pd"; sweep_fail=1; }
    boundary=$k
  fi
done

fail=0
if [ "$SZD" -ne 168 ] || [ "$PZD" -ne 168 ] || ! diff -q "$WORK/souc_zd.txt" "$WORK/py_zd.txt" >/dev/null; then
  echo "MISMATCH (ZD): souc=$SZD python=$PZD"; diff "$WORK/souc_zd.txt" "$WORK/py_zd.txt" | head -20; fail=1
fi
if [ "$SNF" -ne 168 ] || [ "$PNF" -ne 168 ] || ! diff -q "$WORK/souc_nf.txt" "$WORK/py_nf.txt" >/dev/null; then
  echo "MISMATCH (non-Fano): souc=$SNF python=$PNF"; diff "$WORK/souc_nf.txt" "$WORK/py_nf.txt" | head -20; fail=1
fi
if [ "$SAR" -ne 84 ] || [ "$PAR" -ne 84 ] || ! diff -q "$WORK/souc_ar.txt" "$WORK/py_ar.txt" >/dev/null; then
  echo "MISMATCH (dagger bijection): souc=$SAR python=$PAR"; diff "$WORK/souc_ar.txt" "$WORK/py_ar.txt" | head -20; fail=1
fi
if [ "$SME" -ne 2 ] || [ "$PME" -ne 2 ] || ! diff -q "$WORK/souc_me.txt" "$WORK/py_me.txt" >/dev/null; then
  echo "MISMATCH (measure E/Var): souc=$SME python=$PME"; diff "$WORK/souc_me.txt" "$WORK/py_me.txt" | head -20; fail=1
fi
[ "$sweep_fail" -eq 0 ] || fail=1
if [ "$fail" -eq 0 ]; then
  echo "CROSS-VERIFIED: the 168-theorem (structure) + the measure layer (exact + generalized), souc == independent oracle."
  echo "  zero-divisor classes:      $SZD/168 identical pairs"
  echo "  non-Fano triples:          $SNF/168 identical triples"
  echo "  84<->84 dagger bijection:  $SAR/84 identical forward->backward arrows"
  echo "  measure E/Var over Q:      $SME/2 exact rational values identical (on-locus 0/1,0/1; off-locus 0/1,1/150)"
  echo "  generalized sweep:         souc exact for k=1..$boundary; overflow-censored k>$boundary, EXACTLY where"
  echo "                             the unbounded oracle needs BIGINT (i64 exactness boundary located at k=$boundary)."
  echo "  Structure: Lean native_decide-proven (84/336/168, nonFanoCount=168, arrows 84, bridge, dagger)."
  echo "  Measure:   Frente A -- exact Q; E=Var=0 on locus; Var=1/150 off; general sweep 2/(3*10^2k) to the i64 wall."
  exit 0
fi
exit 1
