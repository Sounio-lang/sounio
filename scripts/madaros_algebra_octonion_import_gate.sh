#!/usr/bin/env bash
# Madaros native gate — use algebra::octonion::{oct_mul} under default Madaros.
#
# Acceptance:
#   - Default ./bin/souc (Madaros) compile+run of the import driver
#   - Product e1*e2 → e3: scaled prints 1000 then 0
#   - Sentinel ALGEBRA_OCTONION_IMPORT_OK
#
# Requires a Madaros built AFTER the stack-probe prologue fix
# (self-hosted/native/frame.sio + encode.sio emit_touch_rsp). Checked-in
# bin/madaros-linux-x86_64 may predate the fix; prefer
# artifacts/self-hosted/madaros from build_modular_madaros.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE || true
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-./bin/souc}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== madaros_algebra_octonion_import_gate =="
engine_line="$($SOUC --version 2>&1 | head -1 || true)"
echo "engine: $engine_line"
if echo "$engine_line" | grep -qi lean_single; then
  echo "FAIL: gate must run under default Madaros, not lean_single"
  exit 1
fi

RAW=""
for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}" \
            "$(pwd)/artifacts/self-hosted/madaros" "$(pwd)/bin/madaros-linux-x86_64"; do
  if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null || true)" != '#!' ]]; then
    RAW="$cand"
    break
  fi
done
if [[ -z "$RAW" ]]; then
  echo "MADAROS_ALGEBRA_OCTONION_IMPORT_GATE_BLOCKED reason=no_raw_madaros" >&2
  exit 1
fi
echo "raw_elf=$RAW"
echo "raw_elf_sha256=$(sha256sum "$RAW" | awk '{print $1}')"
echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

SRC=tests/madaros/algebra_octonion_import/oct_mul_import.sio

echo "== compile $SRC =="
if ! $SOUC compile "$SRC" -o "$OUT/oct.elf" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: native compile"
  tail -40 "$OUT/compile.log" || true
  fail=1
else
  echo "PASS: compile"
  echo "== run =="
  set +e
  "$OUT/oct.elf" >"$OUT/run.out" 2>"$OUT/run.err"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAIL: run rc=$rc (SEGV=139 is the historical multi-module oct_mul mode)"
    cat "$OUT/run.err" 2>/dev/null || true
    fail=1
  else
    # Expect: 1000\n0\nALGEBRA_OCTONION_IMPORT_OK
    line1="$(sed -n '1p' "$OUT/run.out" | tr -d '[:space:]')"
    line2="$(sed -n '2p' "$OUT/run.out" | tr -d '[:space:]')"
    echo "out_e3_scaled=$line1 out_e0_scaled=$line2"
    if [[ "$line1" != "1000" ]]; then
      echo "FAIL: e1*e2 e3 component want 1000 got $line1"
      fail=1
    fi
    if [[ "$line2" != "0" ]]; then
      echo "FAIL: real part want 0 got $line2"
      fail=1
    fi
    if ! grep -q 'ALGEBRA_OCTONION_IMPORT_OK' "$OUT/run.out"; then
      echo "FAIL: missing sentinel"
      cat "$OUT/run.out"
      fail=1
    else
      echo "PASS: product + sentinel"
    fi
  fi
fi

# souc run path (wrapper)
if [[ $fail -eq 0 ]]; then
  echo "== souc run =="
  if $SOUC run "$SRC" >"$OUT/srun.out" 2>"$OUT/srun.err"; then
    if grep -q 'ALGEBRA_OCTONION_IMPORT_OK' "$OUT/srun.out"; then
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

echo
if [[ $fail -eq 0 ]]; then
  echo "MADAROS_ALGEBRA_OCTONION_IMPORT_GATE_OK"
  exit 0
fi
echo "MADAROS_ALGEBRA_OCTONION_IMPORT_GATE_FAIL"
exit 1
