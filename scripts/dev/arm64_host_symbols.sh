#!/usr/bin/env bash
# arm64_host_symbols.sh — recover a function->address symbol table for the
# (stripped) self-hosted arm64 compiler, for either the aarch64-linux or
# aarch64-macos build, so gdb/lldb can be pointed at specific compiler functions
# while debugging the arm64-host self-miscompilation (compile_primary reads the
# identifier token-end span `ne` as garbage → var_find fails → mov x0,#0).
#
# How it works: --debug-fn-map makes the compiler dump `fnmap fn=N off=M name=NAME`
# for every function during the arm64 emit (FN_OFF = byte offset into __text).
# The binary maps __text at vaddr TEXT_BASE, so runtime addr = TEXT_BASE + off:
#   aarch64-linux  ELF:    TEXT_BASE = 0x401000
#   aarch64-macos  Mach-O: TEXT_BASE = 0x100004000  (__text section vmaddr)
# The base is auto-detected from the produced binary when otool/readelf is
# available, else falls back to the per-target default above.
#
# Usage:
#   scripts/dev/arm64_host_symbols.sh <souc> [aarch64-linux|aarch64-macos] [TEXT_BASE]
#     → builds the arm64 host with --debug-fn-map and writes <out>.syms (name<TAB>0xADDR)
#   env: ARM64_HOST_OUT (default /tmp/arm64host-dbg[.macho]), ARM64_HOST_SYMS
#
# macOS (native lldb — the fast venue for the hunt):
#   scripts/dev/arm64_host_symbols.sh ./artifacts/self-hosted/souc-self-hosted-arm64-macos aarch64-macos
#   lldb -- /tmp/arm64host-dbg.macho /tmp/var.sio /tmp/o.macho --target aarch64-macos
#   (lldb) br set -a $(grep -P '^compile_primary_a64\t' /tmp/arm64host.syms | cut -f2)
#   (lldb) br set -a $(grep -P '^var_find\t' /tmp/arm64host.syms | cut -f2)
#   (lldb) run     # then step to the identifier-span read; watch `ne`/TE[EP] go wrong
#
# Linux (gdb under qemu — works for targeted breaks; slow to reach main through
# the bundle, so prefer native lldb on Apple Silicon):
#   qemu-aarch64-static -g 2345 /tmp/arm64host-dbg test.sio /tmp/o.elf --target aarch64-linux &
#   gdb-multiarch -ex 'set architecture aarch64' -ex 'target remote :2345' \
#     -ex "break *$(grep -P '^var_find\t' /tmp/arm64host.syms | cut -f2)"
set -euo pipefail

SOUC="${1:?usage: $0 <souc> [aarch64-linux|aarch64-macos] [TEXT_BASE]}"
TARGET="${2:-aarch64-linux}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAN="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
SYMS="${ARM64_HOST_SYMS:-/tmp/arm64host.syms}"

case "$TARGET" in
  aarch64-macos) DEFAULT_BASE="0x100004000"; OUT="${ARM64_HOST_OUT:-/tmp/arm64host-dbg.macho}" ;;
  aarch64-linux) DEFAULT_BASE="0x401000";    OUT="${ARM64_HOST_OUT:-/tmp/arm64host-dbg}" ;;
  *) echo "error: unsupported target $TARGET (use aarch64-linux or aarch64-macos)" >&2; exit 64 ;;
esac

"$SOUC" "$LEAN" "$OUT" --target "$TARGET" --debug-fn-map > /tmp/arm64host-fnmap.txt 2>&1
chmod +x "$OUT" 2>/dev/null || true

# Auto-detect __text vmaddr from the produced binary; fall back to the default.
detect_base() {
  if [ "$TARGET" = "aarch64-macos" ]; then
    local t
    for t in otool llvm-otool; do
      if command -v "$t" >/dev/null 2>&1; then
        "$t" -l "$OUT" 2>/dev/null | awk '/sectname __text/{f=1} f&&$1=="addr"{print $2; exit}'
        return
      fi
    done
  else
    if command -v readelf >/dev/null 2>&1; then
      # first executable LOAD vaddr + 0x1000 (text page); or just trust default
      :
    fi
  fi
}
DETECTED="$(detect_base 2>/dev/null || true)"
TEXT_BASE="${3:-${DETECTED:-$DEFAULT_BASE}}"

python3 - "$TEXT_BASE" <<'PY' > "$SYMS"
import sys
base = int(sys.argv[1], 16)
import re
off = None
for line in open("/tmp/arm64host-fnmap.txt"):
    line = line.rstrip("\n")
    m = re.match(r"^fnmap fn=\d+ off=(\d+) name=(.+)$", line)   # 1-line format
    if m:
        print(f"{m.group(2)}\t{hex(base+int(m.group(1)))}"); continue
    if line.startswith(" off="): off = int(line.split("=",1)[1])  # 3-line format
    elif line.startswith(" name=") and off is not None:
        print(f"{line.split('=',1)[1]}\t{hex(base+off)}"); off = None
PY

echo "wrote $SYMS ($(wc -l < "$SYMS") symbols), host=$OUT target=$TARGET TEXT_BASE=$TEXT_BASE" >&2
grep -P '^(compile_primary|compile_primary_a64|compile_stmt_a64|var_find|emit_load_var_a64|name_eq)\t' "$SYMS" >&2 || true
