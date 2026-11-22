#!/usr/bin/env bash
# heap_limit_source_gate.sh — no entry trampoline derives heap_limit from the
# Linux reservation's upper bound.
#
# nc_heap_arena_bytes() is min(1 TiB, ...) minus the carve-outs: an upper
# bound used only to reject impossible single allocations at compile time.
# The Linux heap_limit is written by emit_entry_heap_reserve_into from the
# length actually mapped; Windows and macOS map a fixed 2 GiB and take
# nc_heap_arena_bytes_fixed_2gib(). A trampoline that loads the upper bound
# into a register sets heap_limit ~1 TiB past a smaller mapping, and arena
# exhaustion becomes a SIGSEGV instead of exit 181 (ADR-028).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CG="$ROOT/self-hosted/native/codegen_x86_linux.sio"
[[ -f "$CG" ]] || { echo "heap_limit_source_gate: FAIL missing $CG"; exit 1; }

bad=$(grep -nF -e 'emit_mov_reg_imm(' "$CG" | grep -F -e 'nc_heap_arena_bytes())' || true)
if [[ -n "$bad" ]]; then
  echo "heap_limit_source_gate: FAIL heap_limit derived from the reservation upper bound:"
  echo "$bad"
  exit 1
fi
# Every Linux trampoline maps through the reserve.
n_linux=$(grep -cE '^(pub )?fn emit_entry_trampoline(_mmap)?_into\(' "$CG")
n_reserve=$(awk '/^(pub )?fn emit_entry_trampoline(_mmap)?_into\(/{f=1} f&&/emit_entry_heap_reserve_into\(nc\)/{c++; f=0} /^}/{f=0} END{print c+0}' "$CG")
if [[ "$n_linux" -ne "$n_reserve" ]]; then
  echo "heap_limit_source_gate: FAIL $n_reserve of $n_linux Linux entry trampolines use emit_entry_heap_reserve_into"
  exit 1
fi
echo "heap_limit_source_gate: PASS ($n_linux Linux trampolines reserve; no heap_limit from the upper bound)"
