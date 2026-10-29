#!/usr/bin/env bash
# SOUNIO_NV2_ARENA_GIB sizes the runtime arena of a Madaros-compiled program.
#
# A Madaros-compiled program keeps every aggregate in one bump arena mapped at
# start-up. It was a fixed 2 GiB, and Madaros compiling itself (the gen3 rung of
# madaros_fixed_point_gate.sh) exhausted it within seconds, while parsing
# main.sio (#2966). The compile can now ask for a larger mapping; the program
# maps it with MAP_NORESERVE, so only touched pages are committed.
#
# The witness threads a 1 MiB struct by value through 1300 calls: about 2.6 GB
# of arena. Built without the variable it must stop with `arena full` (rc 181);
# built with SOUNIO_NV2_ARENA_GIB=4 it must finish and print the right sums.
# 4 GiB also pins the destructure: when `let (b2, v) = b.bump()` copied b2 out
# of the parser temporary __tupN (one more 1 MiB per call, ~3.9 GB) the witness
# stopped at arena full under 4 GiB (#2966).
# A bad value must be reported and fall back to 2 GiB.
#
# Usage: madaros_arena_size_gate.sh   (MADAROS_BIN = compiler under test,
#                                      default bin/souc)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/souc}"
WORK="$(mktemp -d /tmp/sounio-arena-size.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-arena-size] FAIL: $*" >&2; exit 1; }

cat >"$WORK/witness.sio" <<'EOF'
struct Big {
    n: i64,
    data: [i64; 131072],
}

impl Big {
    fn bump(self) -> (Big, i64) with Mut, Panic, Div {
        var s = self
        s.n = s.n + 1
        (s, s.data[7] + s.n)
    }
}

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    var b = Big { n: 0, data: [0; 131072] }
    b.data[7] = 41
    var acc: i64 = 0
    var i: i64 = 0
    while i < 1300 {
        let (b2, v) = b.bump()
        b = b2
        acc = acc + v
        i = i + 1
    }
    println(acc)
    println(b.n)
    return 0
}
EOF

build() {
  local out="$1"; shift
  ( ulimit -s 1572864 2>/dev/null || true; env "$@" "$MADAROS" build "$WORK/witness.sio" "$out" ) >"$out.log" 2>&1 \
    || { tail -n 20 "$out.log" >&2; fail "build failed ($*)"; }
  chmod +x "$out"
}

build "$WORK/default.elf" SOUNIO_NV2_ARENA_GIB=
set +e
"$WORK/default.elf" >"$WORK/default.out" 2>&1
rc_default=$?
set -e
[[ "$rc_default" -eq 181 ]] || fail "default 2 GiB build should stop with arena full (rc 181), got rc=$rc_default: $(head -n 2 "$WORK/default.out")"
grep -q 'arena full' "$WORK/default.out" || fail "default build exited 181 without the arena-full diagnostic"

build "$WORK/big.elf" SOUNIO_NV2_ARENA_GIB=4
set +e
"$WORK/big.elf" >"$WORK/big.out" 2>&1
rc_big=$?
set -e
[[ "$rc_big" -eq 0 ]] || fail "SOUNIO_NV2_ARENA_GIB=4 build should finish, got rc=$rc_big: $(head -n 2 "$WORK/big.out")"
# acc = 1300*41 + (1 + ... + 1300) = 53300 + 845650
expected=$'898950\n1300'
[[ "$(cat "$WORK/big.out")" == "$expected" ]] || fail "SOUNIO_NV2_ARENA_GIB=4 printed $(tr '\n' ' ' <"$WORK/big.out"), expected 898950 1300"

( ulimit -s 1572864 2>/dev/null || true; SOUNIO_NV2_ARENA_GIB=5 "$MADAROS" build "$WORK/witness.sio" "$WORK/bad.elf" ) >"$WORK/bad.log" 2>&1 || true
grep -q 'SOUNIO_NV2_ARENA_GIB=5 is not one of' "$WORK/bad.log" || fail "an unsupported SOUNIO_NV2_ARENA_GIB value was not reported"

echo "[madaros-arena-size] PASS: 2 GiB default stops at arena full (181); SOUNIO_NV2_ARENA_GIB=4 runs the ~2.6 GB witness to 898950/1300; a bad value is reported"
