#!/usr/bin/env bash
# Instruction-level differential test of the Lean model formal/Bootstrap/X86Int.lean
# and decoder formal/Bootstrap/X86Dec.lean against the CPU this runs on.
#
#   bootstrap/model/x86int_diff_gate.sh [cases] [seed]
#   bootstrap/model/x86int_diff_gate.sh <x86diff_cpu binary> [cases] [seed]
#
# x86diff_cpu runs each random case in `int` mode on the CPU; X86IntDiffModel
# decodes the instruction with X86Dec.decode and runs one X86Int.step; the two
# results must agree on rip delta, all 16 registers, CF SF ZF OF and the 64
# memory bytes around rsp. A flag the model leaves undefined ('-') is not
# compared. Any case refused by X86Dec.decode or stuck in X86Int.step fails
# the gate. When cases >= 200000, every form in X86Dec.decode (excluding
# syscall) must occur at least 1000 times.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ "${1:-}" =~ ^[0-9]+$ || -z "${1:-}" ]]; then
  cpu=""
  cases="${1:-200000}"
  seed="${2:-1}"
else
  cpu="$1"
  cases="${2:-200000}"
  seed="${3:-1}"
  if [[ ! -x "$cpu" ]]; then
    echo "X86INT_DIFF_FAIL: CPU binary not executable: $cpu" >&2
    exit 2
  fi
fi

lean="${LEAN_BIN:+$LEAN_BIN/}lean"
lake="${LEAN_BIN:+$LEAN_BIN/}lake"
cxx="${CXX:-g++}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [[ -z "$cpu" ]]; then
  cpu="$work/x86diff_cpu"
  "$cxx" -std=c++23 -O2 -ffreestanding -nostdlib -static -fno-pie -no-pie \
    -fno-stack-protector -fno-exceptions -fno-rtti \
    -o "$cpu" "$root/bootstrap/model/x86diff_cpu.cpp"
fi

"$cpu" "$cases" "$seed" int > "$work/cpu.txt"

model_rc=0
if command -v "$lake" >/dev/null 2>&1; then
  (cd "$root/formal" && "$lake" build x86int-diff >/dev/null)
  "$root/formal/.lake/build/bin/x86int-diff" \
    < "$work/cpu.txt" > "$work/model.txt" 2> "$work/counts.txt" || model_rc=$?
else
  mkdir -p "$work/olean/Bootstrap"
  (
    cd "$root/formal"
    for m in PDist X86Env X86Int X86Dec; do
      LEAN_PATH="$work/olean" "$lean" "Bootstrap/$m.lean" -o "$work/olean/Bootstrap/$m.olean"
    done
    LEAN_PATH="$work/olean" "$lean" --run Bootstrap/X86IntDiffModel.lean
  ) < "$work/cpu.txt" > "$work/model.txt" 2> "$work/counts.txt" || model_rc=$?
fi

cat "$work/counts.txt"

awk -v cases="$cases" -v min_req=1000 '
  /^  form [^:]+: [0-9]+$/ {
    name = $2; sub(/:$/, "", name)
    cnt = $3 + 0
    forms++
    if (cases >= 200000 && cnt < min_req) {
      printf "X86INT_DIFF_COVERAGE_FAIL %s=%d (< %d)\n", name, cnt, min_req
      bad_cov = 1
    }
  }
  END {
    if (forms != 45) {
      printf "X86INT_DIFF_COVERAGE_FAIL expected 45 forms, got %d\n", forms
      exit 1
    }
    if (bad_cov) exit 1
  }' "$work/counts.txt"

cmp_rc=0
awk -v cases="$cases" -v cpu_file="$work/cpu.txt" '
  function cmp(a, b,   i, ca, cb) {
    if (a == b) return 1
    if (length(a) != length(b)) return 0
    for (i = 1; i <= length(a); i++) {
      ca = substr(a, i, 1); cb = substr(b, i, 1)
      if (cb != "-" && ca != cb) return 0
    }
    return 1
  }
  {
    n++
    if ((getline cl < cpu_file) <= 0) { print "cpu EOF at line " n; bad++; next }
    split(cl, c, " \\| "); split($0, m, " \\| ")
    if (c[1] != m[1]) {
      if (bad < 20) print "case mismatch at line " n
      bad++
      next
    }
    if (!cmp(c[2], m[2])) {
      bad++
      if (bad <= 20) {
        print "DIFF " c[1]
        print "  cpu   " c[2]
        print "  model " m[2]
      }
    }
  }
  END {
    close(cpu_file)
    if (n != cases) { print "expected " cases " cases, got " n; exit 1 }
    if (bad) { print "X86INT_DIFF_FAIL " bad " of " n; exit 1 }
    print "X86INT_DIFF_OK " n " cases agree"
  }' "$work/model.txt" || cmp_rc=$?

if [[ $cmp_rc -ne 0 ]]; then
  exit "$cmp_rc"
fi
if [[ $model_rc -ne 0 ]]; then
  echo "X86INT_DIFF_FAIL model exited $model_rc"
  exit "$model_rc"
fi
