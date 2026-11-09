#!/usr/bin/env bash
# SOUNIO_NV2_HANDLE_LOG2 sizes the handle table of a Madaros-compiled program.
#
# Every managed aggregate (over 16 bytes) takes a handle for the life of the
# process. The table held 2^22 = 4,194,304, and the handle-id range could not
# grow past the ELF load base 0x400000: a struct-base word below the capacity
# is a handle id, at or above it a raw pointer. Madaros compiling itself (the
# gen3 rung of madaros_fixed_point_gate.sh) filled it after parsing about 3 MB
# of its 28 MB of source (#2966). A compile can now ask for 2^23 .. 2^27; the
# load base moves to the capacity with it.
#
# The witness keeps a 4.5-million-node Box list alive and sums it.
#   - built without the variable: must stop with `handles full` (rc 182);
#   - built with SOUNIO_NV2_HANDLE_LOG2=23: must finish with the right sum and
#     load at 0x800000;
#   - built with SOUNIO_NV2_HANDLE_LOG2=22: must be byte-identical to the
#     default build;
#   - an unsupported value must be reported and fall back to 22.
#
# Usage: madaros_handle_capacity_gate.sh   (MADAROS_BIN = compiler under test,
#                                           default bin/souc)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/souc}"
WORK="$(mktemp -d /tmp/sounio-handle-capacity.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-handle-capacity] FAIL: $*" >&2; exit 1; }

cat >"$WORK/witness.sio" <<'SIO'
struct Node {
    v: i64,
    w: i64,
    next: Option<Box<Node>>,
}

fn total(head: &Option<Box<Node>>) -> i64 with Mut, Panic, Div {
    var acc: i64 = 0
    var cur = *head
    while true {
        match cur {
            Some(n) => {
                acc = acc + (*n).v
                cur = (*n).next
            }
            None => { break }
        }
    }
    acc
}

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    var head: Option<Box<Node>> = None
    var i: i64 = 0
    while i < 4500000 {
        head = Some(Box::new(Node { v: i, w: 0, next: head }))
        i = i + 1
    }
    println(total(&head))
    return 0
}
SIO

build() {
  local out="$1"; shift
  ( ulimit -s 1572864 2>/dev/null || true; env "$@" "$MADAROS" build "$WORK/witness.sio" "$out" ) >"$out.log" 2>&1 \
    || { tail -n 20 "$out.log" >&2; fail "build failed ($*)"; }
  chmod +x "$out"
}

build "$WORK/default.elf" SOUNIO_NV2_HANDLE_LOG2=
set +e
"$WORK/default.elf" >"$WORK/default.out" 2>&1
rc_default=$?
set -e
[[ "$rc_default" -eq 182 ]] || fail "default build should stop with handles full (rc 182), got rc=$rc_default: $(head -n 2 "$WORK/default.out")"
grep -q 'handles full' "$WORK/default.out" || fail "default build exited 182 without the handles-full diagnostic"

build "$WORK/big.elf" SOUNIO_NV2_HANDLE_LOG2=23
set +e
"$WORK/big.elf" >"$WORK/big.out" 2>&1
rc_big=$?
set -e
# sum 0 .. 4499999
[[ "$rc_big" -eq 0 && "$(cat "$WORK/big.out")" == "10124997750000" ]] \
  || fail "SOUNIO_NV2_HANDLE_LOG2=23 build should print 10124997750000, got rc=$rc_big: $(head -n 2 "$WORK/big.out")"
entry="$(od -An -tx8 -j24 -N8 "$WORK/big.elf" | tr -d ' ')"
(( 16#$entry >= 16#800000 && 16#$entry < 16#1000000 )) \
  || fail "SOUNIO_NV2_HANDLE_LOG2=23 build should load at 0x800000, entry is 0x$entry"

build "$WORK/log22.elf" SOUNIO_NV2_HANDLE_LOG2=22
cmp -s "$WORK/default.elf" "$WORK/log22.elf" || fail "SOUNIO_NV2_HANDLE_LOG2=22 changed the ELF"

( ulimit -s 1572864 2>/dev/null || true; SOUNIO_NV2_HANDLE_LOG2=30 "$MADAROS" build "$WORK/witness.sio" "$WORK/bad.elf" ) >"$WORK/bad.log" 2>&1 || true
grep -q 'SOUNIO_NV2_HANDLE_LOG2=30 is not one of' "$WORK/bad.log" || fail "an unsupported SOUNIO_NV2_HANDLE_LOG2 value was not reported"

echo "[madaros-handle-capacity] PASS: default stops at handles full (182); SOUNIO_NV2_HANDLE_LOG2=23 keeps 4.5M nodes live, loads at 0x800000 and sums to 10124997750000; LOG2=22 is byte-identical; a bad value is reported"
