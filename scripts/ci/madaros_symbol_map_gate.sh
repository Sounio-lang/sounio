#!/usr/bin/env bash
# SOUNIO_NV2_SYMBOL_MAP writes `<entry vaddr hex> <name>` per function and
# leaves the ELF byte-identical.
#
# The native ELF has no section headers, so a crash address in a
# Madaros-built program names no function. madaros_fixed_point_gate.sh asks
# for this map next to gen2, so a gen3 crash site can be named from the
# artifact (#2966). The witness: helper_beta calls helper_alpha and main calls
# helper_beta; the mapped addresses must be the targets of those calls.
#
# Usage: madaros_symbol_map_gate.sh   (MADAROS_BIN = compiler under test,
#                                      default bin/souc)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/souc}"
WORK="$(mktemp -d /tmp/sounio-symbol-map.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-symbol-map] FAIL: $*" >&2; exit 1; }
command -v objdump >/dev/null || fail "objdump is required"

cat >"$WORK/w.sio" <<'EOF'
fn helper_alpha(x: i64) -> i64 with Mut, Panic, Div { x * 3 + 1 }
fn helper_beta(x: i64) -> i64 with Mut, Panic, Div { helper_alpha(x) - 2 }
fn main() -> i32 with IO, Mut, Panic, Div {
    println(helper_beta(5))
    return 0
}
EOF
build() { ( ulimit -s 1572864 2>/dev/null || true; env "$@" ) >>"$WORK/build.log" 2>&1 || { tail -n 20 "$WORK/build.log" >&2; fail "build failed"; }; }
build SOUNIO_NV2_SYMBOL_MAP="$WORK/w.map" "$MADAROS" build "$WORK/w.sio" "$WORK/with.elf"
build SOUNIO_NV2_SYMBOL_MAP= "$MADAROS" build "$WORK/w.sio" "$WORK/without.elf"
cmp -s "$WORK/with.elf" "$WORK/without.elf" || fail "the ELF differs when a symbol map is requested"
chmod +x "$WORK/with.elf"
[[ "$("$WORK/with.elf")" == "14" ]] || fail "witness did not print 14 (helper_beta(5) = 3*5+1-2)"
[[ -s "$WORK/w.map" ]] || fail "no symbol map written"
addr() { awk -v n="$1" '$2 == n { print $1 }' "$WORK/w.map"; }
a="$(addr helper_alpha)"; b="$(addr helper_beta)"; m="$(addr main)"
[[ -n "$a" && -n "$b" && -n "$m" ]] || fail "map lacks main/helper_alpha/helper_beta: $(head -n 5 "$WORK/w.map" | tr '\n' ';')"
calls_from() {  # calls_from <start hex> : call targets in the function starting there
  objdump -D -b binary -m i386:x86-64 --adjust-vma=0x400000 \
    --start-address="0x$1" --stop-address="$(printf '0x%x' $((16#$1 + 512)))" "$WORK/with.elf" \
    | grep -oE 'call +0x[0-9a-f]+' | awk '{print $2}' | sed 's/^0x//'
}
calls_from "$b" | grep -qx "$a" || fail "helper_beta (0x$b) does not call helper_alpha's mapped address 0x$a"
calls_from "$m" | grep -qx "$b" || fail "main (0x$m) does not call helper_beta's mapped address 0x$b"
echo "[madaros-symbol-map] PASS: map names main/helper_beta/helper_alpha at their call targets; ELF byte-identical with and without the map"
