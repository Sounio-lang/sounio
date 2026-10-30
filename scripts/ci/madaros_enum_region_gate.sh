#!/usr/bin/env bash
# A function returning a payload-free enum must reclaim its temporaries.
#
# Region reclamation (native/gc.sio) resets the arena at a function's return
# when nothing it allocated escaped. lowerer_rgn_return_kind_mut decided which
# return types qualify, and rgn_copy_cost refused every enum name, so a function
# returning a plain enum never opened a region. The parser's
# pt_keyword_lookup_span (-> TokenKind) therefore left two 2080-byte arrays in
# the arena on every token-kind read, and Madaros compiling itself ran out of
# arena while parsing (#2966).
#
# The witness calls a `-> Kind` function 1.5 million times; each call fills a
# 256-word scratch array (2080 bytes of arena). Without the region that is
# about 3 GB and the 2 GiB arena fills (rc 181); with it the program finishes
# and counts 500000 of each kind.
#
# Usage: madaros_enum_region_gate.sh   (MADAROS_BIN = compiler under test,
#                                       default bin/souc)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/souc}"
WORK="$(mktemp -d /tmp/sounio-enum-region.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-enum-region] FAIL: $*" >&2; exit 1; }

cat >"$WORK/witness.sio" <<'EOF'
enum Kind { Low, Mid, High }

fn classify(n: i64) -> Kind with Mut, Panic, Div {
    var scratch: [i64; 256] = [0; 256]
    scratch[n % 256] = n
    let v = scratch[n % 256] % 3
    if v == 0 { return Kind::Low }
    if v == 1 { return Kind::Mid }
    Kind::High
}

fn main() -> i32 with IO, Mut, Panic, Div {
    var low: i64 = 0
    var mid: i64 = 0
    var high: i64 = 0
    var i: i64 = 0
    while i < 1500000 {
        let k = classify(i)
        match k {
            Kind::Low => { low = low + 1 }
            Kind::Mid => { mid = mid + 1 }
            Kind::High => { high = high + 1 }
        }
        i = i + 1
    }
    println(low)
    println(mid)
    println(high)
    return 0
}
EOF

( ulimit -s 1572864 2>/dev/null || true; env SOUNIO_NV2_ARENA_GIB= "$MADAROS" build "$WORK/witness.sio" "$WORK/w.elf" ) >"$WORK/build.log" 2>&1 \
  || { tail -n 20 "$WORK/build.log" >&2; fail "build failed"; }
chmod +x "$WORK/w.elf"
set +e
"$WORK/w.elf" >"$WORK/out" 2>&1
rc=$?
set -e
[[ "$rc" -eq 0 ]] || fail "witness exited rc=$rc (181 = arena full: the enum-returning function kept its temporaries): $(head -n 1 "$WORK/out")"
expected=$'500000\n500000\n500000'
[[ "$(cat "$WORK/out")" == "$expected" ]] || fail "witness printed $(tr '\n' ' ' <"$WORK/out"), expected 500000 x3"
echo "[madaros-enum-region] PASS: 1.5M calls of a -> Kind function with a 2080-byte scratch each run in the default 2 GiB arena"
