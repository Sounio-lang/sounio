#!/usr/bin/env bash
# Process-environment differential test of formal/Bootstrap/X86Env.lean,
# formal/Bootstrap/X86Int.lean (syscall), and formal/Bootstrap/X86Dec.lean
# against the Linux kernel and x86-64 CPU this runs on.
#
#   bootstrap/model/x86env_diff_gate.sh [scenarios] [seed]
#   bootstrap/model/x86env_diff_gate.sh <x86env_diff_cpu binary> [scenarios] [seed]
#
# x86env_diff_cpu executes randomized multi-step syscall scenarios (0f 05 in
# isolated fork() children) and static-nostdlib execve stack probes;
# X86EnvDiffModel decodes 0f 05 with X86Dec.decode, steps X86Int.step /
# X86Env.syscall, checks X86Env.Mem.wf preservation at every step, and verifies
# X86Env.initStack on every execve case. Undefined register r11 after syscall
# is written as '-' by the model and skipped during comparison.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ "${1:-}" =~ ^[0-9]+$ || -z "${1:-}" ]]; then
  cpu=""
  scenarios="${1:-2000}"
  seed="${2:-1}"
else
  cpu="$1"
  scenarios="${2:-2000}"
  seed="${3:-1}"
  if [[ ! -x "$cpu" ]]; then
    echo "X86ENV_DIFF_FAIL: CPU binary not executable: $cpu" >&2
    exit 2
  fi
fi

if ! command -v "${LEAN_BIN:+$LEAN_BIN/}lean" >/dev/null 2>&1 && [[ -x "$HOME/.elan/bin/lean" ]]; then
  export PATH="$HOME/.elan/bin:$PATH"
fi

lean="${LEAN_BIN:+$LEAN_BIN/}lean"
lake="${LEAN_BIN:+$LEAN_BIN/}lake"
cxx="${CXX:-g++}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

probe="$work/x86env_stack_probe"
"$cxx" -std=c++23 -O2 -DINIT_STACK_PROBE -ffreestanding -nostdlib -static \
  -fno-pie -no-pie -fno-stack-protector -fno-exceptions -fno-rtti \
  -o "$probe" "$root/bootstrap/model/x86env_diff_cpu.cpp"

if [[ -z "$cpu" ]]; then
  cpu="$work/x86env_diff_cpu"
  "$cxx" -std=c++23 -O2 -Wall -Wextra \
    -o "$cpu" "$root/bootstrap/model/x86env_diff_cpu.cpp"
fi

"$cpu" "$scenarios" "$seed" "$probe" > "$work/cpu.txt"

model_rc=0
if command -v "$lake" >/dev/null 2>&1 && grep -q 'x86env-diff' "$root/formal/lakefile.lean"; then
  (cd "$root/formal" && "$lake" build x86env-diff >/dev/null)
  "$root/formal/.lake/build/bin/x86env-diff" \
    < "$work/cpu.txt" > "$work/model.txt" 2> "$work/counts.txt" || model_rc=$?
else
  mkdir -p "$work/olean/Bootstrap"
  (
    cd "$root/formal"
    for m in PDist X86Env X86Int X86Dec; do
      LEAN_PATH="$work/olean" "$lean" "Bootstrap/$m.lean" -o "$work/olean/Bootstrap/$m.olean" >/dev/null
    done
  )
  (
    cd "$root/formal"
    LEAN_PATH="$work/olean" "$lean" --run Bootstrap/X86EnvDiffModel.lean
  ) < "$work/cpu.txt" > "$work/model.txt" 2> "$work/counts.txt" || model_rc=$?
fi

cat "$work/counts.txt"

awk -v scenarios="$scenarios" '
  /^  category [^:]+: [0-9]+$/ {
    name = $2; sub(/:$/, "", name)
    cnt = $3 + 0
    cats++
    if (cnt < 1) {
      printf "X86ENV_DIFF_COVERAGE_FAIL %s=%d (< 1)\n", name, cnt
      bad_cov = 1
    }
    if (name == "init_stack" && scenarios >= 2000 && cnt < 200) {
      printf "X86ENV_DIFF_COVERAGE_FAIL init_stack=%d (< 200)\n", cnt
      bad_cov = 1
    }
  }
  END {
    if (cats != 18) {
      printf "X86ENV_DIFF_COVERAGE_FAIL expected 18 categories, got %d\n", cats
      exit 1
    }
    if (bad_cov) exit 1
  }' "$work/counts.txt"

cmp_rc=0
awk -v scenarios="$scenarios" -v cpu_file="$work/cpu.txt" '
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
        print "DIFF line " n
        print "  cpu   " c[2]
        print "  model " m[2]
      }
    }
  }
  END {
    close(cpu_file)
    if (n < scenarios) { print "expected at least " scenarios " lines, got " n; exit 1 }
    if (bad) { print "X86ENV_DIFF_FAIL " bad " of " n; exit 1 }
    print "X86ENV_DIFF_OK " n " lines agree"
  }' "$work/model.txt" || cmp_rc=$?

if [[ $cmp_rc -ne 0 ]]; then
  exit "$cmp_rc"
fi
if [[ $model_rc -ne 0 ]]; then
  echo "X86ENV_DIFF_FAIL model exited $model_rc"
  exit "$model_rc"
fi
