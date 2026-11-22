#!/usr/bin/env bash
# Instruction-level differential test of formal/Bootstrap/AArch64Min.lean
# against the local Apple Silicon CPU via mmap(..., MAP_JIT).
#
#   bootstrap/model/aarch64_diff_gate.sh [cases] [seed]
#   bootstrap/model/aarch64_diff_gate.sh <aarch64_diff_cpu binary> [cases] [seed]
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
    echo "AARCH64_DIFF_FAIL: CPU binary not executable: $cpu" >&2
    exit 2
  fi
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "AARCH64_DIFF_FAIL: requires Darwin arm64 (Apple Silicon), got $(uname -s) $(uname -m)" >&2
  exit 2
fi

if [[ -n "${LEAN_BIN:-}" ]]; then
  lean="$LEAN_BIN/lean"
  lake="$LEAN_BIN/lake"
elif command -v lean >/dev/null 2>&1; then
  lean="$(command -v lean)"
  lake="$(command -v lake || true)"
elif [[ -x "$HOME/.elan/bin/lean" ]]; then
  lean="$HOME/.elan/bin/lean"
  lake="$HOME/.elan/bin/lake"
else
  echo "AARCH64_DIFF_FAIL: lean not found" >&2
  exit 2
fi

if [[ -z "${CXX:-}" ]]; then
  if [[ -x "/usr/bin/clang++" ]]; then
    cxx="/usr/bin/clang++"
  else
    cxx="clang++"
  fi
else
  cxx="$CXX"
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [[ -z "$cpu" ]]; then
  cpu="$work/aarch64_diff_cpu"
  "$cxx" -std=c++23 -O2 -o "$cpu" "$root/bootstrap/model/aarch64_diff_cpu.cpp"
fi

"$cpu" "$cases" "$seed" > "$work/cpu.txt"

model_rc=0
if [[ -n "$lake" ]] && grep -q 'aarch64-diff' "$root/formal/lakefile.lean" 2>/dev/null; then
  (cd "$root/formal" && "$lake" build aarch64-diff >/dev/null)
  "$root/formal/.lake/build/bin/aarch64-diff" \
    < "$work/cpu.txt" > "$work/model.txt" 2> "$work/counts.txt" || model_rc=$?
else
  mkdir -p "$work/olean/Bootstrap" "$work/c"
  prefix="$("$lean" --print-prefix)"
  leanc="$prefix/bin/leanc"
  (
    cd "$root/formal"
    LEAN_PATH="$work/olean" "$lean" "Bootstrap/AArch64Min.lean" \
      -o "$work/olean/Bootstrap/AArch64Min.olean" -c "$work/c/AArch64Min.c"
    LEAN_PATH="$work/olean" "$lean" "Bootstrap/AArch64DiffModel.lean" \
      -o "$work/olean/Bootstrap/AArch64DiffModel.olean" -c "$work/c/AArch64DiffModel.c"
  )
  if [[ -x "$leanc" ]]; then
    "$leanc" -O2 -o "$work/aarch64-diff" "$work/c/AArch64Min.c" "$work/c/AArch64DiffModel.c"
    "$work/aarch64-diff" < "$work/cpu.txt" > "$work/model.txt" 2> "$work/counts.txt" || model_rc=$?
  else
    (cd "$root/formal" && LEAN_PATH="$work/olean" "$lean" --run Bootstrap/AArch64DiffModel.lean) \
      < "$work/cpu.txt" > "$work/model.txt" 2> "$work/counts.txt" || model_rc=$?
  fi
fi

cat "$work/counts.txt"

awk -v cases="$cases" -v min_req=1000 '
  /^  form [^:]+: [0-9]+$/ {
    name = $2; sub(/:$/, "", name)
    cnt = $3 + 0
    forms++
    if (cases >= 200000 && cnt < min_req) {
      printf "AARCH64_DIFF_COVERAGE_FAIL %s=%d (< %d)\n", name, cnt, min_req
      bad_cov = 1
    }
  }
  END {
    if (forms != 32) {
      printf "AARCH64_DIFF_COVERAGE_FAIL expected 32 forms, got %d\n", forms
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
    if (bad) { print "AARCH64_DIFF_FAIL " bad " of " n; exit 1 }
    print "AARCH64_DIFF_OK " n " cases agree"
  }' "$work/model.txt" || cmp_rc=$?

if [[ $cmp_rc -ne 0 ]]; then
  exit "$cmp_rc"
fi
if [[ $model_rc -ne 0 ]]; then
  echo "AARCH64_DIFF_FAIL: model exited with $model_rc" >&2
  exit "$model_rc"
fi
