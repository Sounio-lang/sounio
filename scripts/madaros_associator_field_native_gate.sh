#!/usr/bin/env bash
# Madaros native gate — use algebra::associator_field under default Madaros.
#
# Acceptance:
#   - Default ./bin/souc (Madaros) compile+run of the import driver
#   - Non-Fano (e1,e2,e4): ‖assoc‖²=4, g2=2, augmented var=4.25
#   - Pentagon (e1,e2,e4,e1): variance=0.96
#   - Sentinel ASSOCIATOR_FIELD_NATIVE_OK
#   - L0 run-pass tests (associator_field_octonion + pentagon) ALL PASS
#
# Depends on algebra::octonion::oct_mul lo/hi frame split (#1274). No Madaros
# rebuild required for this stdlib visibility fix.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE || true
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-./bin/souc}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== madaros_associator_field_native_gate =="
engine_line="$($SOUC --version 2>&1 | head -1 || true)"
echo "engine: $engine_line"
if echo "$engine_line" | grep -qi lean_single; then
  echo "FAIL: gate must run under default Madaros, not lean_single"
  exit 1
fi
if ! echo "$engine_line" | grep -qi Madaros; then
  echo "WARN: version string does not mention Madaros: $engine_line"
fi
echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

SRC=tests/madaros/associator_field_native/associator_field_import.sio

echo "== compile $SRC =="
if ! $SOUC compile "$SRC" -o "$OUT/af.elf" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: native compile"
  tail -40 "$OUT/compile.log" || true
  fail=1
else
  echo "PASS: compile"
  echo "== run =="
  set +e
  "$OUT/af.elf" >"$OUT/run.out" 2>"$OUT/run.err"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAIL: run rc=$rc (SEGV=139 was the historical exclusive-ref multi-mod mode)"
    cat "$OUT/run.err" 2>/dev/null || true
    fail=1
  else
    # Expect: 4000\n2000\n4250\n960000\nASSOCIATOR_FIELD_NATIVE_OK
    line1="$(sed -n '1p' "$OUT/run.out" | tr -d '[:space:]')"
    line2="$(sed -n '2p' "$OUT/run.out" | tr -d '[:space:]')"
    line3="$(sed -n '3p' "$OUT/run.out" | tr -d '[:space:]')"
    line4="$(sed -n '4p' "$OUT/run.out" | tr -d '[:space:]')"
    echo "norm_sq_x1000=$line1 g2_x1000=$line2 aug_x1000=$line3 pent_var_x1e6=$line4"
    if [[ "$line1" != "4000" ]]; then
      echo "FAIL: non-Fano norm_sq want 4000 got $line1"
      fail=1
    fi
    if [[ "$line2" != "2000" ]]; then
      echo "FAIL: g2_residual want 2000 got $line2"
      fail=1
    fi
    if [[ "$line3" != "4250" ]]; then
      echo "FAIL: augmented variance want 4250 got $line3"
      fail=1
    fi
    if [[ "$line4" != "960000" ]]; then
      echo "FAIL: pentagon variance want 960000 got $line4"
      fail=1
    fi
    if ! grep -q 'ASSOCIATOR_FIELD_NATIVE_OK' "$OUT/run.out"; then
      echo "FAIL: missing sentinel"
      cat "$OUT/run.out"
      fail=1
    else
      echo "PASS: numeric sentinels + ASSOCIATOR_FIELD_NATIVE_OK"
    fi
  fi
fi

# souc run path (wrapper)
if [[ $fail -eq 0 ]]; then
  echo "== souc run =="
  if $SOUC run "$SRC" >"$OUT/srun.out" 2>"$OUT/srun.err"; then
    if grep -q 'ASSOCIATOR_FIELD_NATIVE_OK' "$OUT/srun.out"; then
      echo "PASS: souc run"
    else
      echo "FAIL: souc run missing sentinel"
      cat "$OUT/srun.out"
      fail=1
    fi
  else
    echo "FAIL: souc run"
    tail -20 "$OUT/srun.err" || true
    fail=1
  fi
fi

# L0 acceptance surfaces (full window suite)
if [[ $fail -eq 0 ]]; then
  for t in tests/run-pass/associator_field_octonion.sio tests/run-pass/associator_field_pentagon.sio; do
    echo "== L0 $t =="
    if $SOUC compile "$t" -o "$OUT/l0.elf" >"$OUT/l0.compile" 2>&1; then
      set +e
      "$OUT/l0.elf" >"$OUT/l0.out" 2>"$OUT/l0.err"
      lrc=$?
      set -e
      if [[ $lrc -ne 0 ]]; then
        echo "FAIL: L0 run rc=$lrc"
        cat "$OUT/l0.err" 2>/dev/null || true
        fail=1
        break
      fi
      if ! grep -q 'ALL PASS' "$OUT/l0.out"; then
        echo "FAIL: L0 missing ALL PASS"
        cat "$OUT/l0.out"
        fail=1
        break
      fi
      echo "PASS: ALL PASS"
    else
      echo "FAIL: L0 compile"
      tail -20 "$OUT/l0.compile" || true
      fail=1
      break
    fi
  done
fi

echo
if [[ $fail -eq 0 ]]; then
  echo "MADAROS_ASSOCIATOR_FIELD_NATIVE_GATE_OK"
  exit 0
fi
echo "MADAROS_ASSOCIATOR_FIELD_NATIVE_GATE_FAIL"
exit 1
