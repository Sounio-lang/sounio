#!/usr/bin/env bash
# Hardware fault-injection gate for the canonical 38-byte x86-64 machine-code
# TMR voter (formal/Bootstrap/TMRVoter.lean) and exact Rat bound verifier
# (formal/Bootstrap/TMRBoundCheck.lean).
#
#   bootstrap/model/tmr_fault_gate.sh [trials] [seed]
#
# Compiles bootstrap/model/tmr_fault_inject.cpp, executes the single-bit
# exhaustive sweep + Mode A (clean voter) + Mode B (4-site faulty voter) on
# the host x86-64 CPU, and verifies in Lean via exact Rat arithmetic that
# single-fault SDC == 0 and every sweep point satisfies 3*q_body^2 + 9*r_total.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"

trials="${1:-20000}"
seed="${2:-1}"

lean="${LEAN_BIN:+$LEAN_BIN/}lean"
lake="${LEAN_BIN:+$LEAN_BIN/}lake"
if ! command -v "$lean" >/dev/null 2>&1 && [[ -x "$HOME/.elan/bin/lean" ]]; then
  lean="$HOME/.elan/bin/lean"
  lake="$HOME/.elan/bin/lake"
fi
cxx="${CXX:-g++}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

inj="$work/tmr_fault_inject"
"$cxx" -std=c++23 -O2 \
  -o "$inj" "$root/bootstrap/model/tmr_fault_inject.cpp"

"$inj" "$trials" "$seed" > "$work/inject.txt"

if command -v "$lake" >/dev/null 2>&1 && grep -q 'tmr-bound-check' "$root/formal/lakefile.lean"; then
  (cd "$root/formal" && "$lake" build tmr-bound-check >/dev/null)
  "$root/formal/.lake/build/bin/tmr-bound-check" < "$work/inject.txt"
else
  mkdir -p "$work/olean/Bootstrap"
  (
    cd "$root/formal"
    for m in PDist TMR TMRExposure X86Env X86Int X86Dec; do
      LEAN_PATH="$work/olean" "$lean" "Bootstrap/$m.lean" -o "$work/olean/Bootstrap/$m.olean" >/dev/null
    done
    LEAN_PATH="$work/olean" "$lean" --run Bootstrap/TMRBoundCheck.lean < "$work/inject.txt"
  )
fi
