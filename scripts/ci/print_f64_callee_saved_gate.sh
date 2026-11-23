#!/usr/bin/env bash
# print_f64 keeps the SysV callee-saved registers (rbx, r12-r15) on all eight
# exits of tests/abi/print_f64_callee_saved.sio, checked under ptrace by
# scripts/ci/print_f64_callee_saved_probe.cpp. Uses ./bin/souc, so the caller
# selects the compiler (CI: the verified current-source Madaros).
# Measured on the SIO-396 branch before its fix: rbx clobbered on 8/8 calls,
# r12 and r13 on the two |x| >= 2^63 calls.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${1:-$(mktemp -d "${TMPDIR:-/tmp}/print-f64-callee-saved.XXXXXX")}"
mkdir -p "$WORK"
case "$WORK" in /*) ;; *) WORK="$PWD/$WORK" ;; esac
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror \
  scripts/ci/print_f64_callee_saved_probe.cpp -o "$WORK/probe"
"$ROOT/bin/souc" compile tests/abi/print_f64_callee_saved.sio -o "$WORK/witness" > "$WORK/build.log" 2>&1
[[ -s "$WORK/witness" ]] || { echo 'print_f64 ABI witness executable missing' >&2; cat "$WORK/build.log" >&2; exit 1; }
chmod +x "$WORK/witness"
timeout 60 "$WORK/probe" "$WORK/witness" 8 > "$WORK/stdout.log" 2> "$WORK/probe.log" || { cat "$WORK/probe.log" >&2; exit 1; }
cat "$WORK/probe.log"
printf '%s\n' '10000000000000000000.000000' '-1000000000000000019884624838656.000000' \
  '1.500000' '-2.250000' '0.000000' 'nan' 'inf' '-inf' > "$WORK/expected-stdout.log"
cmp "$WORK/expected-stdout.log" "$WORK/stdout.log" || { echo 'print_f64 ABI witness stdout differs' >&2; exit 1; }
grep -Fxq 'PRINT_F64_CALLEE_SAVED_PASS calls=8' "$WORK/probe.log"
echo PRINT_F64_CALLEE_SAVED_GATE_PASS
