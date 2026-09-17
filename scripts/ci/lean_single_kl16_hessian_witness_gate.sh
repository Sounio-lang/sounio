#!/usr/bin/env bash
# KL-16: pin lean_single Hessian witnesses (Tier 1–3 + KL-16b ch4–7).
#
# Engine: SOUNIO_SOUC_ENGINE=lean_single (committed seed via bin/souc).
#
# Green corpus (runtime values, not just typecheck):
#   Tier 1  tests/run-pass/epistemic_hessian_of.sio
#           tests/run-pass/epistemic_hessian_transcendentals.sio
#   Tier 2  tests/run-pass/epistemic_hessian_8inputs.sio
#           H[4,5] of a product is analytic 1.0 (KL-16b closed the
#           VAR_HSHADOW leak across Knowledge locals).
#   Tier 3  tests/run-pass/epistemic_hessian_two_arg.sio   (atan2 / pow, ch 0–3)
#   KL-16b  tests/run-pass/epistemic_hessian_ch47.sio      (unary+two-arg ch 4–7)
#
# Residual (still OPEN — not asserted here):
#   - multi-channel interproc (KL-16c closed FO ch0 + H[0,0] only)
#   - loop accumulation; if/else merge of shadow slots
#   - a64 atan2/pow AD (value-only today)
# See also: scripts/ci/lean_single_kl16c_interproc_shadow_gate.sh
#
# println(f64) is __native_print_f64_n(_, 6); expected lines use that format.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

case "$(uname -s 2>/dev/null || echo unknown)/$(uname -m 2>/dev/null || echo unknown)" in
  Linux/x86_64|Linux/amd64) ;;
  *)
    echo "[kl16-hessian] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
SOUC="${SOUC:-$ROOT/bin/souc}"

OUT="$(mktemp -d /tmp/sounio-kl16-hessian.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

echo "== lean_single_kl16_hessian_witness_gate =="
echo "engine: SOUNIO_SOUC_ENGINE=$SOUNIO_SOUC_ENGINE"
echo "souc:   $SOUC"

fails=0

# run_witness <label> <src> <expected multiline stdout>
run_witness() {
  local label="$1" src="$2" expect="$3"
  local log="$OUT/$label.log" err="$OUT/$label.err"
  local rc=0
  set +e
  "$SOUC" run "$src" >"$log" 2>"$err"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "FAIL  $label -- run rc=$rc"
    sed 's/^/        /' "$err" "$log" || true
    fails=$((fails + 1))
    return
  fi
  if ! diff -u <(printf '%s\n' "$expect") "$log" >"$OUT/$label.diff"; then
    echo "FAIL  $label -- stdout mismatch"
    sed 's/^/        /' "$OUT/$label.diff" || true
    fails=$((fails + 1))
    return
  fi
  echo "ok    $label"
}

run_witness of \
  tests/run-pass/epistemic_hessian_of.sio \
  $'1.000000\n2.000000\n10.000000'

run_witness transcendentals \
  tests/run-pass/epistemic_hessian_transcendentals.sio \
  $'-0.500000\n0.000000\n0.500000\n0.000000\n1.000000\n0.000000'

run_witness eight_inputs \
  tests/run-pass/epistemic_hessian_8inputs.sio \
  $'1.000000\n1.000000\n1.000000\n2.000000'

run_witness two_arg \
  tests/run-pass/epistemic_hessian_two_arg.sio \
  $'0.000000\n-1.000000\n2.000000\n0.000000'

run_witness ch47 \
  tests/run-pass/epistemic_hessian_ch47.sio \
  $'-0.500000\n1.000000\n-1.000000\n2.000000'

if [[ "$fails" -ne 0 ]]; then
  echo "LEAN_SINGLE_KL16_HESSIAN_WITNESS_GATE_FAIL ($fails)"
  exit 1
fi
echo "LEAN_SINGLE_KL16_HESSIAN_WITNESS_GATE_OK"
