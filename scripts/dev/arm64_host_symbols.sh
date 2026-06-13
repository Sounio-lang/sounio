#!/usr/bin/env bash
# arm64_host_symbols.sh — recover a function->address symbol table for the
# (stripped) self-hosted arm64 compiler, so gdb/lldb can be pointed at specific
# compiler functions while debugging the arm64-host self-miscompilation.
#
# How it works: --debug-fn-map makes the compiler dump `fnmap fn=N off=M name=NAME`
# for every function during the arm64 emit (FN_OFF = byte offset into __text).
# The arm64-linux ELF maps __text at vaddr TEXT_BASE (0x401000), so the runtime
# address of a function is TEXT_BASE + off.
#
# Usage:
#   scripts/dev/arm64_host_symbols.sh <x86-souc-with-debug-fn-map> [TEXT_BASE]
#     → builds /tmp/arm64host-dbg and writes /tmp/arm64host.syms (name<TAB>0xADDR)
#
# The <x86-souc-with-debug-fn-map> must be an x86 souc built from a tree where
# compile_all_arm64 contains the --debug-fn-map dump (committed 2026-06-13).
#
# gdb (Linux, under qemu — slow for the full bundle, ok for targeted breaks):
#   qemu-aarch64-static -g 2345 /tmp/arm64host-dbg test.sio /tmp/o.elf --target aarch64-linux &
#   gdb-multiarch -ex 'set architecture aarch64' -ex 'target remote :2345' \
#     -ex "break *$(grep -P '^var_find\t' /tmp/arm64host.syms | cut -f2)"
#
# lldb (macOS, native arm64 — fast; preferred for the final hunt):
#   # build the arm64-macos host with --debug-fn-map, derive TEXT_BASE from the
#   # Mach-O __text vmaddr, regenerate the syms, then:
#   lldb -- ./souc-arm64-macos test.sio /tmp/o.elf --target aarch64-macos
#   (lldb) br set -a <addr_of_compile_primary_a64>
set -euo pipefail

SOUC="${1:?usage: $0 <x86-souc-with-debug-fn-map> [TEXT_BASE]}"
TEXT_BASE="${2:-0x401000}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAN="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
OUT="${ARM64_HOST_OUT:-/tmp/arm64host-dbg}"
SYMS="${ARM64_HOST_SYMS:-/tmp/arm64host.syms}"

"$SOUC" "$LEAN" "$OUT" --target aarch64-linux --debug-fn-map > /tmp/arm64host-fnmap.txt 2>&1
chmod +x "$OUT"

python3 - "$TEXT_BASE" <<'PY' > "$SYMS"
import sys
base = int(sys.argv[1], 16)
off = fn = name = None
for line in open("/tmp/arm64host-fnmap.txt"):
    line = line.rstrip("\n")
    if line.startswith("fnmap fn="): fn = line.split("=",1)[1]
    elif line.startswith(" off="): off = int(line.split("=",1)[1])
    elif line.startswith(" name="):
        name = line.split("=",1)[1]
        print(f"{name}\t{hex(base+off)}")
PY

echo "wrote $SYMS ($(wc -l < "$SYMS") symbols), host $OUT, TEXT_BASE=$TEXT_BASE" >&2
grep -P '^(compile_primary|compile_primary_a64|compile_stmt_a64|var_find|emit_load_var_a64|name_eq)\t' "$SYMS" >&2 || true
