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

# The Python oracle emits all six faces (PAIR/TRIPLE/ARROW/MEASURE/SWEEP/BIG).
python3 scripts/research/verify_zd168_oracle.py > "$WORK/py_all.txt"

echo "[face 1/9] zero-divisor census: souc vs oracle ..."
./bin/souc run tests/run-pass/sedenion_zd_census_168.sio 2>/dev/null | grep '^PAIR ' | sort -u > "$WORK/souc_zd.txt"
grep '^PAIR ' "$WORK/py_all.txt" | sort -u > "$WORK/py_zd.txt"
SZD=$(wc -l < "$WORK/souc_zd.txt"); PZD=$(wc -l < "$WORK/py_zd.txt")

echo "[face 2/9] non-Fano census: souc vs oracle ..."
./bin/souc run tests/run-pass/octonion_nonfano_census_168.sio 2>/dev/null | grep '^TRIPLE ' | sort -u > "$WORK/souc_nf.txt"
grep '^TRIPLE ' "$WORK/py_all.txt" | sort -u > "$WORK/py_nf.txt"
SNF=$(wc -l < "$WORK/souc_nf.txt"); PNF=$(wc -l < "$WORK/py_nf.txt")

echo "[face 3/9] 84<->84 dagger bijection map: souc vs oracle ..."
./bin/souc run tests/run-pass/octonion_dagger_bijection_84.sio 2>/dev/null | grep '^ARROW ' | sort -u > "$WORK/souc_ar.txt"
grep '^ARROW ' "$WORK/py_all.txt" | sort -u > "$WORK/py_ar.txt"
SAR=$(wc -l < "$WORK/souc_ar.txt"); PAR=$(wc -l < "$WORK/py_ar.txt")

echo "[face 4/9] measure-layer exact E/Var over Q: souc vs Python fractions ..."
./bin/souc run tests/run-pass/sedenion_measure_annihilation_exact.sio 2>/dev/null | grep '^MEASURE ' | sort > "$WORK/souc_me.txt"
grep '^MEASURE ' "$WORK/py_all.txt" | sort > "$WORK/py_me.txt"
SME=$(wc -l < "$WORK/souc_me.txt"); PME=$(wc -l < "$WORK/py_me.txt")

echo "[face 5/9] generalized sweep + i64 boundary: souc (overflow-censored) vs unbounded oracle ..."
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

echo "[face 6/9] UNBOUNDED bigint sweep (past the i64 wall): souc BigNat vs unbounded oracle ..."
./bin/souc run tests/run-pass/sedenion_measure_annihilation_bigint.sio 2>/dev/null | grep '^BIG ' | sort -n -k2 > "$WORK/souc_big.txt"
grep '^BIG ' "$WORK/py_all.txt" | sort -n -k2 > "$WORK/py_big.txt"
SBIG=$(wc -l < "$WORK/souc_big.txt"); PBIG=$(wc -l < "$WORK/py_big.txt")

echo "[face 7/9] full bigint (add/sub/mul/divmod/gcd/cmp/eq/is_zero): souc vs Python big-int oracle ..."
: > "$WORK/souc_bn.txt"
for f in bignat_selftest bignat_selftest_divmod_rem bignat_selftest_signed bignat_selftest_eq_true bignat_selftest_eq_false bignat_selftest_iszero_true bignat_selftest_iszero_false; do
  ./bin/souc run "tests/run-pass/$f.sio" 2>/dev/null | grep '^OP ' >> "$WORK/souc_bn.txt"
done
sort -o "$WORK/souc_bn.txt" "$WORK/souc_bn.txt"
python3 scripts/research/bignat_oracle.py 2>/dev/null | grep '^OP ' | sort > "$WORK/py_bn.txt"
SBN=$(wc -l < "$WORK/souc_bn.txt"); PBN=$(wc -l < "$WORK/py_bn.txt")

echo "[face 8/9] rational-CD channel over unbounded Q: souc vs Python fractions oracle ..."
: > "$WORK/souc_rb.txt"
for f in sedenion_ratbig_channel_case1 sedenion_ratbig_channel_case2 sedenion_ratbig_channel_case3; do
  ./bin/souc run "tests/run-pass/$f.sio" 2>/dev/null | grep -E '^(CASE|R5|R12) ' >> "$WORK/souc_rb.txt"
done
sort -o "$WORK/souc_rb.txt" "$WORK/souc_rb.txt"
python3 scripts/research/ratbig_oracle.py 2>/dev/null | grep -E '^(CASE|R5|R12) ' | sort > "$WORK/py_rb.txt"
SRB=$(wc -l < "$WORK/souc_rb.txt"); PRB=$(wc -l < "$WORK/py_rb.txt")

echo "[face 9/9] GENERAL 16-component CD product over Q (all 16 comps, arbitrary rational pair): souc vs oracle ..."
./bin/souc run tests/run-pass/sedenion_cd_full16_q.sio 2>/dev/null | grep -E '^(COMP|DEN) ' | sort > "$WORK/souc_16.txt"
python3 scripts/research/cd16_oracle.py 2>/dev/null | grep -E '^(COMP|DEN) ' | sort > "$WORK/py_16.txt"
S16=$(wc -l < "$WORK/souc_16.txt"); P16=$(wc -l < "$WORK/py_16.txt")

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
if [ "$SBIG" -ne 20 ] || [ "$PBIG" -ne 20 ] || ! diff -q "$WORK/souc_big.txt" "$WORK/py_big.txt" >/dev/null; then
  echo "MISMATCH (bigint sweep): souc=$SBIG python=$PBIG"; diff "$WORK/souc_big.txt" "$WORK/py_big.txt" | head -20; fail=1
fi
if [ "$SBN" -ne 17 ] || [ "$PBN" -ne 17 ] || ! diff -q "$WORK/souc_bn.txt" "$WORK/py_bn.txt" >/dev/null; then
  echo "MISMATCH (bigint ops): souc=$SBN python=$PBN"; diff "$WORK/souc_bn.txt" "$WORK/py_bn.txt" | head -20; fail=1
fi
if [ "$SRB" -ne 9 ] || [ "$PRB" -ne 9 ] || ! diff -q "$WORK/souc_rb.txt" "$WORK/py_rb.txt" >/dev/null; then
  echo "MISMATCH (rational-CD channel): souc=$SRB python=$PRB"; diff "$WORK/souc_rb.txt" "$WORK/py_rb.txt" | head -20; fail=1
fi
if [ "$S16" -ne 34 ] || [ "$P16" -ne 34 ] || ! diff -q "$WORK/souc_16.txt" "$WORK/py_16.txt" >/dev/null; then
  echo "MISMATCH (full-16 CD over Q): souc=$S16 python=$P16"; diff "$WORK/souc_16.txt" "$WORK/py_16.txt" | head -20; fail=1
fi
if [ "$fail" -eq 0 ]; then
  echo "CROSS-VERIFIED (9 faces): 168-theorem structure + measure layer (exact->generalized->UNBOUNDED bigint) + full bigint + rational-CD channel + GENERAL 16-comp CD over Q, souc == independent oracle."
  echo "  zero-divisor classes:      $SZD/168 identical pairs"
  echo "  non-Fano triples:          $SNF/168 identical triples"
  echo "  84<->84 dagger bijection:  $SAR/84 identical forward->backward arrows"
  echo "  measure E/Var over Q:      $SME/2 exact rational values identical (on-locus 0/1,0/1; off-locus 0/1,1/150)"
  echo "  generalized sweep:         souc i64 exact k=1..$boundary; censored k>$boundary where oracle needs BIGINT."
  echo "  UNBOUNDED bigint sweep:    $SBIG/20 exact rational values identical PAST the i64 wall (to 1.5e40)."
  echo "  full bigint ops:           $SBN/17 exact (add/sub/mul/divmod/gcd/cmp/eq/is_zero, to ~10^60)."
  echo "  rational-CD channel:       $SRB/9 exact reduced rationals (locus annihilation over unbounded Q + off-locus)."
  echo "  GENERAL 16-comp CD over Q: $S16/34 exact (all 16 comps, arbitrary rational pair; common-denom circumvents the [Rational;16] wall)."
  echo "  Structure: Lean native_decide-proven. Measure: exact Q, i64 boundary located AND removed via a from-scratch BigInt."
  exit 0
fi
exit 1
