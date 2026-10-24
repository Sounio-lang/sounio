#!/usr/bin/env bash
# [i8; N] / [u8; N] arrays are allocated one byte per element (#2966).
#
# A Madaros-compiled program kept every array element in an 8-byte word, so a
# Name ([i8; 384]) was a 3104-byte object. Parsing main.sio left 1.84 GB of
# managed objects, 95 % of it in [i8; 384] and [i8; 256] buffers, and gen3 grew
# by about 550 MB/s until the runner swapped. Array literals declared as byte
# arrays (a struct field or an annotated `let`) are now packed, copies keep the
# layout of their source, and every element access and str_from_bytes /
# write_file read a header tag to pick the layout.
#
#   1. tests/run-pass/packed_byte_array_semantics.sio must print
#      PACKED_BYTES_OK: reads, writes, sign and zero extension, copies,
#      by-value and by-reference passing, &! mutation, str_from_bytes.
#   2. A witness keeps 700,000 Name-like structs alive in the default 2 GiB
#      arena. Word layout needs about 2.2 GB and stops at arena full (rc 181);
#      packed it needs about 0.3 GB and prints the checksum.
#   3. tests/run-pass/first_global_array_by_ref.sio indexes the first global
#      (at BSS_BASE_LINUX) by reference. The tag read sits one header below the
#      array, which for that global is unmapped; it must print
#      FIRST_GLOBAL_ARRAY_REF_OK, not die with SIGSEGV (stdlib_deep_eigen).
#
# Usage: madaros_packed_bytes_gate.sh   (MADAROS_BIN = compiler under test,
#                                        default bin/souc)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/souc}"
WORK="$(mktemp -d /tmp/sounio-packed-bytes.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-packed-bytes] FAIL: $*" >&2; exit 1; }

build() {
  local src="$1" out="$2"
  ( ulimit -s 1572864 2>/dev/null || true; "$MADAROS" build "$src" "$out" ) >"$out.log" 2>&1 \
    || { tail -n 20 "$out.log" >&2; fail "build failed: $src"; }
  chmod +x "$out"
}

build "$ROOT_DIR/tests/run-pass/packed_byte_array_semantics.sio" "$WORK/sem.elf"
set +e
sem_out="$("$WORK/sem.elf" 2>&1)"; sem_rc=$?
set -e
[[ "$sem_rc" -eq 0 && "$sem_out" == "PACKED_BYTES_OK" ]] || fail "semantics: rc=$sem_rc output=$(tr '\n' ' ' <<<"$sem_out")"

build "$ROOT_DIR/tests/run-pass/first_global_array_by_ref.sio" "$WORK/first.elf"
set +e
first_out="$("$WORK/first.elf" 2>&1)"; first_rc=$?
set -e
[[ "$first_rc" -eq 0 && "$first_out" == "FIRST_GLOBAL_ARRAY_REF_OK" ]] || fail "first global by reference: rc=$first_rc output=$(tr '\n' ' ' <<<"$first_out")"

cat >"$WORK/mem.sio" <<'SIO'
struct Nm {
    buf: [i8; 384],
    len: i64,
}

struct Cell {
    n: Nm,
    next: Option<Box<Cell>>,
}

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    var head: Option<Box<Cell>> = None
    var i: i64 = 0
    while i < 700000 {
        var nm = Nm { buf: [0; 384], len: i }
        nm.buf[0] = (i % 100) as i8
        head = Some(Box::new(Cell { n: nm, next: head }))
        i = i + 1
    }
    var acc: i64 = 0
    var cur = head
    while true {
        match cur {
            Some(c) => {
                acc = acc + (*c).n.buf[0] as i64
                cur = (*c).next
            }
            None => { break }
        }
    }
    println(acc)
    return 0
}
SIO
build "$WORK/mem.sio" "$WORK/mem.elf"
set +e
mem_out="$("$WORK/mem.elf" 2>&1)"; mem_rc=$?
set -e
# sum over i < 700000 of i % 100 = 7000 * (0 + ... + 99) = 34650000
[[ "$mem_rc" -eq 0 && "$mem_out" == "34650000" ]] \
  || fail "700k Name-like structs in 2 GiB: rc=$mem_rc output=$(head -n 2 <<<"$mem_out" | tr '\n' ' ') (181 = arena full: buffers not packed)"

echo "[madaros-packed-bytes] PASS: byte-array semantics hold; the first global indexes by reference; 700k Name-like structs fit the default 2 GiB arena and sum to 34650000"
