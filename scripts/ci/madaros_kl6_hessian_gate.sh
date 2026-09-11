#!/usr/bin/env bash
# KL-6: hessian_of on Madaros -- quotient rule and the composite chain rule.
#
# Three witnesses, one verdict. Each prints its rows and its own sentinel:
#   tests/run-pass/madaros_hessian_quotient.sio        H(a/b), H(1/b)
#   tests/run-pass/madaros_hessian_transcendental.sio  H(sin/cos/exp), leaf and
#                                                      compound inner (kind 6)
#   tests/run-pass/kl6_hessian_composite.sio           f'(g)H(g) on the kind-8
#                                                      path (log, sqrt), and a
#                                                      quotient with a composite
#                                                      numerator
#
# A zero Hessian is not neutral in this algebra: it is the claim "no
# second-order dependence". So the gate also proves it can die. With the
# declared sabotage hooks armed (lower.sio: sab_quotient_hessian,
# sab_hessian_chain) the quotient and composite witnesses must FAIL; a gate that
# stays green with the rule removed is measuring nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
unset SOUNIO_SABOTAGE_QUOTIENT_HESSIAN SOUNIO_SABOTAGE_HESSIAN_CHAIN || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_kl6_hessian_gate =="
echo "compiler under test: $SOUC ${SOUNIO_MADAROS_BIN:+(SOUNIO_MADAROS_BIN=$SOUNIO_MADAROS_BIN)}"

fails=0

# run_witness <label> <src> <sentinel> <expect: pass|fail> [ENV=1 ...]
run_witness() {
  local label="$1" src="$2" sentinel="$3" expect="$4"; shift 4
  local elf="$OUT/$label.elf" log="$OUT/$label.log"
  if ! env "$@" "$SOUC" compile "$src" -o "$elf" >"$OUT/$label.compile.log" 2>&1; then
    echo "FAIL  $label -- compile"
    tail -30 "$OUT/$label.compile.log" || true
    fails=$((fails + 1)); return
  fi
  chmod +x "$elf"
  set +e
  "$elf" >"$log" 2>&1
  local rc=$?
  set -e
  local got=absent
  grep -q "$sentinel" "$log" && got=present
  case "$expect" in
    pass)
      if [[ "$rc" -eq 0 && "$got" == present ]]; then
        echo "ok    $label -- $sentinel (rc=0)"
        sed 's/^/        /' "$log"
      else
        echo "FAIL  $label -- rc=$rc sentinel=$got"
        sed 's/^/        /' "$log" || true
        fails=$((fails + 1))
      fi ;;
    fail)
      # Sabotage control: the sentinel must NOT appear. rc is not enough on its
      # own -- a crash would also be non-zero -- so both are checked.
      if [[ "$got" == absent ]]; then
        echo "ok    $label -- sabotage kills it (rc=$rc, $sentinel absent)"
      else
        echo "FAIL  $label -- sabotage armed and $sentinel still printed: the gate cannot die"
        sed 's/^/        /' "$log" || true
        fails=$((fails + 1))
      fi ;;
  esac
}

run_witness quotient       tests/run-pass/madaros_hessian_quotient.sio       MADAROS_HESSIAN_QUOTIENT_PASS       pass
run_witness transcendental tests/run-pass/madaros_hessian_transcendental.sio MADAROS_HESSIAN_TRANSCENDENTAL_PASS pass
run_witness composite      tests/run-pass/kl6_hessian_composite.sio          KL6_HESSIAN_COMPOSITE_PASS          pass

# Sabotage controls (compile-time hooks, read by lower.sio).
run_witness quotient_sab   tests/run-pass/madaros_hessian_quotient.sio       MADAROS_HESSIAN_QUOTIENT_PASS       fail SOUNIO_SABOTAGE_QUOTIENT_HESSIAN=1
run_witness composite_sab  tests/run-pass/kl6_hessian_composite.sio          KL6_HESSIAN_COMPOSITE_PASS          fail SOUNIO_SABOTAGE_HESSIAN_CHAIN=1

if [[ "$fails" -ne 0 ]]; then
  echo "MADAROS_KL6_HESSIAN_GATE_FAIL ($fails)"
  exit 1
fi
echo "MADAROS_KL6_HESSIAN_GATE_OK"
