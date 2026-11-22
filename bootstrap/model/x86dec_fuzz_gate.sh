#!/usr/bin/env bash
# Hamming-1/2 (`X86Dec.altsAt`) and adversarial byte-boundary differential fuzz
# gate for `formal/Bootstrap/X86Dec.lean` and `formal/Bootstrap/X86Int.lean`
# against the host x86-64 CPU.
#
#   bootstrap/model/x86dec_fuzz_gate.sh [valid_seeds] [adv_count] [seed]
#
# 1. Runs `X86DecFuzzModel gen` to test all Hamming-1 bit-flip neighbours
#    (`X86Dec.altsAt`, verifying 100% agreement with per-bit `decode (flipBit ...)`)
#    of `valid_seeds` (default 5,000) random valid stage-1 instructions (~180,000+
#    Hamming-1 candidates) + `10 * valid_seeds` (50,000) Hamming-2 candidates +
#    `adv_count` (default 100,000) adversarial/random 1..15 byte sequences.
# 2. Executes every accepted non-syscall byte sequence natively on the CPU via
#    `x86dec_fuzz_cpu` right against an unmapped `PROT_NONE` guard page at
#    `0x100000000` (`RFLAGS.TF = 1`), catching any CPU over-read or under-read
#    and comparing all 16 GPRs, CF/SF/ZF/OF flags, rip delta, and 64 memory
#    bytes against `X86Int.step`.
# 3. Verifies the final report via `X86DecFuzzModel verify`.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"

valid_seeds="${1:-5000}"
adv_count="${2:-100000}"
seed="${3:-1}"

lean="${LEAN_BIN:+$LEAN_BIN/}lean"
lake="${LEAN_BIN:+$LEAN_BIN/}lake"
if ! command -v "$lean" >/dev/null 2>&1 && [[ -x "$HOME/.elan/bin/lean" ]]; then
  lean="$HOME/.elan/bin/lean"
  lake="$HOME/.elan/bin/lake"
fi
cxx="${CXX:-g++}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cpu="$work/x86dec_fuzz_cpu"
"$cxx" -std=c++23 -O2 \
  -o "$cpu" "$root/bootstrap/model/x86dec_fuzz_cpu.cpp"

if command -v "$lake" >/dev/null 2>&1 && grep -q 'x86dec-fuzz' "$root/formal/lakefile.lean"; then
  (cd "$root/formal" && "$lake" build x86dec-fuzz >/dev/null)
  "$root/formal/.lake/build/bin/x86dec-fuzz" \
    gen "$valid_seeds" "$adv_count" "$seed" "$work/cases.txt" "$work/stats.txt"
  cpu_rc=0
  "$cpu" "$seed" "$work/cases.txt" > "$work/cpu.txt" || cpu_rc=$?
  "$root/formal/.lake/build/bin/x86dec-fuzz" \
    verify "$work/cpu.txt" "$work/stats.txt"
  if [[ $cpu_rc -ne 0 ]]; then
    exit "$cpu_rc"
  fi
else
  mkdir -p "$work/olean/Bootstrap"
  (
    cd "$root/formal"
    for m in PDist X86Env X86Int X86Dec; do
      LEAN_PATH="$work/olean" "$lean" "Bootstrap/$m.lean" -o "$work/olean/Bootstrap/$m.olean" >/dev/null
    done
    LEAN_PATH="$work/olean" "$lean" --run Bootstrap/X86DecFuzzModel.lean \
      gen "$valid_seeds" "$adv_count" "$seed" "$work/cases.txt" "$work/stats.txt"
  )
  cpu_rc=0
  "$cpu" "$seed" "$work/cases.txt" > "$work/cpu.txt" || cpu_rc=$?
  (
    cd "$root/formal"
    LEAN_PATH="$work/olean" "$lean" --run Bootstrap/X86DecFuzzModel.lean \
      verify "$work/cpu.txt" "$work/stats.txt"
  )
  if [[ $cpu_rc -ne 0 ]]; then
    exit "$cpu_rc"
  fi
fi
