#!/usr/bin/env bash
# Madaros native gate — epistemic::propagate delta-method + MC under default Madaros.
#
# Acceptance:
#   - Default ./bin/souc (Madaros) compile+run of the multi-module witness prints
#     product (6 / 0.25) and exp (literal name; e / e²·0.01) as exact scaled
#     micro-units, plus MC identity / square within numeric bands.
#   - TRUST sentinel PROPAGATE_TRUST_OK from free-function Epistemic path.
#   - Requires a Madaros binary that includes the Wave6 C codegen fix
#     (empty-stub builtins only) so call-site `exp` is not float-hijacked.
#
# Does NOT claim lean_single is Madaros.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# Refuse to silently run lean_single — this gate is Madaros-only.
engine_line="$($SOUC --version 2>&1 | head -1 || true)"
if echo "$engine_line" | grep -qi lean_single; then
  echo "FAIL: gate must run under default Madaros, not lean_single ($engine_line)"
  exit 1
fi
if ! echo "$engine_line" | grep -qi Madaros; then
  echo "WARN: version string does not mention Madaros: $engine_line"
fi
echo "engine: $engine_line"

fail=0

in_band() {
  # $1 value $2 lo $3 hi (all integers)
  local v="$1" lo="$2" hi="$3"
  if [ "$v" -ge "$lo" ] && [ "$v" -le "$hi" ]; then
    return 0
  fi
  return 1
}

echo "== witness_import_propagate (scaled micro-units) =="
if $SOUC compile tests/epistemic_trust/witness_import_propagate.sio -o "$OUT/w.elf" >"$OUT/w.compile" 2>&1; then
  mapfile -t lines < <("$OUT/w.elf" | tr -d '\r')
  p_val="$(echo "${lines[0]:-}" | tr -d '[:space:]')"
  p_var="$(echo "${lines[1]:-}" | tr -d '[:space:]')"
  e_val="$(echo "${lines[2]:-}" | tr -d '[:space:]')"
  e_var="$(echo "${lines[3]:-}" | tr -d '[:space:]')"
  id_val="$(echo "${lines[4]:-}" | tr -d '[:space:]')"
  id_var="$(echo "${lines[5]:-}" | tr -d '[:space:]')"
  sq_val="$(echo "${lines[6]:-}" | tr -d '[:space:]')"
  sq_var="$(echo "${lines[7]:-}" | tr -d '[:space:]')"
  echo "product=($p_val,$p_var) exp=($e_val,$e_var) mc_id=($id_val,$id_var) mc_sq=($sq_val,$sq_var)"

  if [ "$p_val" = "6000000" ] && [ "$p_var" = "250000" ]; then
    echo "PASS: product 6.0 / 0.25"
  else
    echo "FAIL: product sentinels (want 6000000 / 250000)"
    fail=1
  fi

  # exp(1) series ≈ 2.718281 → 2718281; var e²*0.01 ≈ 0.073890 → 73890 (±2)
  # Witness imports the symbol literally named `exp` (not exp_delta).
  if in_band "$e_val" 2718200 2718360 && in_band "$e_var" 73800 74000; then
    echo "PASS: exp e / e²·0.01 (scaled $e_val / $e_var)"
  else
    echo "FAIL: exp sentinels (got $e_val / $e_var)"
    fail=1
  fi

  # MC identity: mean ≈ 1.0 (±5%), var ≈ 0.01 (±50% band for N=5000 CLT)
  if in_band "$id_val" 950000 1050000 && in_band "$id_var" 5000 15000; then
    echo "PASS: MC identity mean≈1 var≈0.01 (scaled $id_val / $id_var)"
  else
    echo "FAIL: MC identity band (got $id_val / $id_var)"
    fail=1
  fi

  # MC square: E[X²]≈4.01, Var≈0.16 for X~N(2,0.01)
  if in_band "$sq_val" 3900000 4120000 && in_band "$sq_var" 80000 250000; then
    echo "PASS: MC square E[X²]≈4.01 var≈0.16 (scaled $sq_val / $sq_var)"
  else
    echo "FAIL: MC square band (got $sq_val / $sq_var)"
    fail=1
  fi
else
  echo "FAIL: native compile"
  tail -30 "$OUT/w.compile"
  fail=1
fi

echo "== propagate_trust sentinel =="
if $SOUC compile tests/epistemic_trust/propagate_trust.sio -o "$OUT/t.elf" >"$OUT/t.compile" 2>&1; then
  if "$OUT/t.elf" | grep -q 'PROPAGATE_TRUST_OK'; then
    echo "PASS: PROPAGATE_TRUST_OK"
  else
    echo "FAIL: sentinel missing (exit=$("$OUT/t.elf" >/dev/null; echo $?))"
    "$OUT/t.elf" || true
    fail=1
  fi
else
  echo "FAIL: trust compile"
  tail -30 "$OUT/t.compile"
  fail=1
fi

echo "== souc run trust driver =="
if $SOUC run tests/epistemic_trust/propagate_trust.sio >"$OUT/run.out" 2>"$OUT/run.err"; then
  if grep -q 'PROPAGATE_TRUST_OK' "$OUT/run.out"; then
    echo "PASS: souc run trust"
  else
    echo "FAIL: souc run missing sentinel"
    cat "$OUT/run.out"
    fail=1
  fi
else
  echo "FAIL: souc run trust"
  tail -30 "$OUT/run.err"
  fail=1
fi

echo
if [ $fail -eq 0 ]; then
  echo "MADAROS_PROPAGATE_NATIVE_GATE_OK"
  exit 0
fi
echo "MADAROS_PROPAGATE_NATIVE_GATE_FAIL"
exit 1
