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
#   4. Packed byte globals carry an object header (global_byte_array_header.sio:
#      by reference, str_from_bytes), and write_file from one writes
#      exactly its bytes (gen3 wrote the 160 MB NC_BIG_ELF this way and read
#      every 8th byte off the end of BSS).
#   5. `x as i8` is the sign-extended low byte (cast_i8_sign_extends.sio), so
#      `CODE[i] == (0x89 as i8)` holds; gen2 compared opcode bytes that way and
#      left every relocation in gen3 unpatched.
#   6. A global array passed by value to a user function (`a: [T; N]`, word,
#      f64 and u8 elements, a method parameter, a relay through a second
#      call) reads the global from its first element (global_array_byval_arg.sio,
#      SIO-279); it read 4 words late.
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

build "$ROOT_DIR/tests/run-pass/global_byte_array_header.sio" "$WORK/gbh.elf"
set +e
gbh_out="$("$WORK/gbh.elf" 2>&1)"; gbh_rc=$?
set -e
[[ "$gbh_rc" -eq 0 && "$gbh_out" == "GLOBAL_BYTE_ARRAY_HEADER_OK" ]] || fail "packed byte globals by reference / str_from_bytes: rc=$gbh_rc output=$(tr '\n' ' ' <<<"$gbh_out")"

build "$ROOT_DIR/tests/run-pass/cast_i8_sign_extends.sio" "$WORK/ci8.elf"
set +e
ci8_out="$("$WORK/ci8.elf" 2>&1)"; ci8_rc=$?
set -e
[[ "$ci8_rc" -eq 0 && "$ci8_out" == "CAST_I8_OK" ]] || fail "x as i8 must be the sign-extended low byte: rc=$ci8_rc output=$(tr '\n' ' ' <<<"$ci8_out")"

build "$ROOT_DIR/tests/run-pass/global_array_byval_arg.sio" "$WORK/gab.elf"
set +e
gab_out="$("$WORK/gab.elf" 2>&1)"; gab_rc=$?
set -e
[[ "$gab_rc" -eq 0 && "$gab_out" == "GLOBAL_ARRAY_BYVAL_OK" ]] || fail "a global array passed by value must read from its first element: rc=$gab_rc output=$(tr '\n' ' ' <<<"$gab_out")"

# write_file from a packed byte global: the gen3 path (NC_BIG_ELF). The file
# must hold exactly the bytes written, not every 8th one.
cat >"$WORK/wf.sio" <<SIO
var PAD: [i64; 4] = [0; 4]
var OUTBUF: [i8; 4096] = [0; 4096]

fn main() -> i32 with IO, Mut, Panic, Div {
    var i: i64 = 0
    while i < 26 {
        OUTBUF[i as usize] = (65 + i) as i8
        i = i + 1
    }
    let n = write_file("$WORK/wf.bin", OUTBUF, 26)
    print_int(n)
    println("")
    return 0
}
SIO
build "$WORK/wf.sio" "$WORK/wf.elf"
set +e
wf_out="$("$WORK/wf.elf" 2>&1)"; wf_rc=$?
set -e
[[ "$wf_rc" -eq 0 && "$wf_out" == "26" && "$(cat "$WORK/wf.bin" 2>/dev/null)" == "ABCDEFGHIJKLMNOPQRSTUVWXYZ" ]] \
  || fail "write_file from a packed byte global: rc=$wf_rc output=$wf_out file=$(od -An -c "$WORK/wf.bin" 2>/dev/null | head -2 | tr '\n' ' ')"

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

echo "[madaros-packed-bytes] PASS: byte-array semantics hold; the first global indexes by reference; packed byte globals carry a header (by reference, str_from_bytes, write_file); x as i8 sign-extends; a global array passed by value reads from element 0; 700k Name-like structs fit the default 2 GiB arena and sum to 34650000"
