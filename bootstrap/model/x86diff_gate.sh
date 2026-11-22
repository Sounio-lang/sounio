#!/usr/bin/env bash
# Instruction-level differential test of the Lean model formal/Bootstrap/X86Min.lean
# against the CPU this runs on.
#
#   bootstrap/model/x86diff_gate.sh <x86diff_cpu binary> [cases] [seed]
#
# x86diff_cpu runs each random case on the CPU; X86DiffModel runs the same case on
# the model; the two results must agree on rip, all 16 registers, CF SF ZF OF and
# the 32 stack bytes around rsp. A flag the model leaves undefined ('-') is not
# compared. Needs `lean` (the toolchain in formal/lean-toolchain) on PATH or in
# $LEAN_BIN, and an x86-64 Linux CPU.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cpu="${1:?usage: x86diff_gate.sh <x86diff_cpu> [cases] [seed]}"
cases="${2:-200000}"
seed="${3:-1}"
lean="${LEAN_BIN:+$LEAN_BIN/}lean"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

"$root/bootstrap/model/build_model.sh" "$work/olean"
"$cpu" "$cases" "$seed" > "$work/cpu.txt"
(cd "$root/formal" && LEAN_PATH="$work/olean" "$lean" --run Bootstrap/X86DiffModel.lean) \
  < "$work/cpu.txt" > "$work/model.txt"

awk -v cases="$cases" '
  function cmp(a, b,   i, ca, cb) {
    if (length(a) != length(b)) return 0
    for (i = 1; i <= length(a); i++) {
      ca = substr(a, i, 1); cb = substr(b, i, 1)
      if (cb != "-" && ca != cb) return 0
    }
    return 1
  }
  NR == FNR { cpu[FNR] = $0; next }
  {
    n++
    split(cpu[FNR], c, " \\| "); split($0, m, " \\| ")
    if (c[1] != m[1]) { print "case mismatch at line " FNR; bad++; next }
    if (!cmp(c[2], m[2])) {
      bad++
      if (bad <= 20) { print "DIFF " c[1]; print "  cpu   " c[2]; print "  model " m[2] }
    }
  }
  END {
    if (n != cases) { print "expected " cases " cases, got " n; exit 1 }
    if (bad) { print "X86DIFF_FAIL " bad " of " n; exit 1 }
    print "X86DIFF_OK " n " cases agree"
  }' "$work/cpu.txt" "$work/model.txt"
