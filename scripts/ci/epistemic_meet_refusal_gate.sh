#!/usr/bin/env bash
# ADR-016 meet refuses at run time, with a diagnostic, instead of fusing.
#
# meet(...) fuses measured values by generalised least squares over their GUM
# covariance. Its two refusals depend on measured values, so they are run-time
# contracts (ADR-016, decision 3.6): the program prints the refusal, the sources
# with their uncertainties and the statistic, and stops without producing the
# fused value. A run-pass test cannot express "must stop", so this gate does.
#
#   refuse_inconsistent   E_n = 1.06 > 1 for two independent sources
#   accept_policy         the same pair under `consistency: 1.1` fuses
#   refuse_peelle         correlated pair, E_n policy 20: negative GLS weight
#   refuse_peelle_birge   the same pair under meet_birge: still refused
#
# Usage: epistemic_meet_refusal_gate.sh   (SOUNIO_MADAROS_BIN = compiler under
#                                          test, default bin/souc)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUNIO_MADAROS_BIN:-$ROOT/bin/souc}"
DIR="$ROOT/tests/epistemic-meet"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail() { echo "[epistemic-meet-refusal] FAIL: $*" >&2; exit 1; }

build() { # build <name>
  ( ulimit -s unlimited 2>/dev/null || true
    "$SOUC" compile "$DIR/$1.sio" -o "$OUT/$1.elf" ) >"$OUT/$1.build.log" 2>&1 \
    || { tail -n 30 "$OUT/$1.build.log" >&2; fail "compile $1"; }
  chmod +x "$OUT/$1.elf"
}
run() { # run <name> -> RC, LOG
  set +e; "$OUT/$1.elf" >"$OUT/$1.run.log" 2>&1; RC=$?; set -e
  LOG="$OUT/$1.run.log"
}
expect_refusal() { # expect_refusal <name> <reason substring> <statistic substring>
  build "$1"
  run "$1"
  [[ "$RC" -ne 0 ]] || { cat "$LOG" >&2; fail "$1 exited 0"; }
  grep -q NOT_REACHED "$LOG" && { cat "$LOG" >&2; fail "$1 used the fused value"; }
  grep -qF "error: meet refused: $2" "$LOG" || { cat "$LOG" >&2; fail "$1: missing reason '$2'"; }
  grep -qF "$3" "$LOG" || { cat "$LOG" >&2; fail "$1: missing statistic '$3'"; }
  grep -qF "  source " "$LOG" || { cat "$LOG" >&2; fail "$1: sources not reported"; }
  echo "  $1: refused (rc=$RC)"
}

echo "== epistemic_meet_refusal_gate =="
expect_refusal refuse_inconsistent "two sources are inconsistent above the E_n policy" "E_n(a, b) = 1.06066"
expect_refusal refuse_peelle "a generalised-least-squares weight is negative (Peelle)" "GLS weight(tip4) = -0.8497"
expect_refusal refuse_peelle_birge "a generalised-least-squares weight is negative (Peelle)" "GLS weight(tip4) = -0.8497"
build accept_policy
run accept_policy
[[ "$RC" -eq 0 ]] && grep -q MEET_POLICY_ACCEPT_OK "$LOG" || { cat "$LOG" >&2; fail "accept_policy rc=$RC"; }
echo "  accept_policy: fused under consistency: 1.1"
echo "[epistemic-meet-refusal] PASS"
