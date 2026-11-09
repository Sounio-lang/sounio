#!/usr/bin/env bash
# A compile that exhausts the IR name-id pool REFUSES; it does not emit a binary
# with blanked symbol names.
#
# ir_intern_name hands out one id per distinct name (functions, fields, types,
# mangled methods). The pool holds 262144 ids; when it is full the interner
# returns 0, every nid-keyed lowering table then silently misses, and the old
# behaviour was a warning followed by an ELF that exited 0. ir_arena_report_violation
# now treats the overflow as fatal at the one barrier every build route passes
# before the ELF is written. A real >262144-name program is impractical, so the
# driver honours a TEST-ONLY soft limit, SOUNIO_IR_NAME_POOL_LIMIT=<n>, which
# lowers the id cap to min(n, 262144).
#
#   1. With SOUNIO_IR_NAME_POOL_LIMIT=8 a small program (a few functions, a
#      struct, a println) must fail to build: non-zero exit, a diagnostic that
#      contains "IR name-id pool overflowed" and "refusing", and NO output file.
#   2. Without the variable the same program builds (rc 0) and prints
#      NAME_POOL_OK 42, so the limit is inert unless asked for.
#
# Usage: madaros_name_pool_overflow_gate.sh   (MADAROS_BIN = compiler under test,
#                                              default bin/souc)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/souc}"
WORK="$(mktemp -d /tmp/sounio-name-pool-overflow.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-name-pool-overflow] FAIL: $*" >&2; exit 1; }

cat >"$WORK/prog.sio" <<'SIO'
struct Pair {
    left: i64,
    right: i64,
}

fn make_pair(a: i64, b: i64) -> Pair {
    Pair { left: a, right: b }
}

fn sum_pair(p: Pair) -> i64 {
    p.left + p.right
}

fn main() -> i32 with IO {
    let p = make_pair(40, 2)
    print("NAME_POOL_OK ")
    println(sum_pair(p))
    return 0
}
SIO

# (a) tiny id cap: must refuse -- classified, not judged by exit status alone:
# rc 1..123, the capacity diagnostic present, and no artifact written.
# shellcheck source=../lib/classify_compile.sh
source "$ROOT_DIR/scripts/lib/classify_compile.sh"
sounio_capture_rc limited_rc "$WORK/limited.log" -- bash -c '
  ulimit -s 1572864 2>/dev/null || true
  SOUNIO_IR_NAME_POOL_LIMIT=8 exec "$1" build "$2" "$3"' _ "$MADAROS" "$WORK/prog.sio" "$WORK/limited.elf"
sounio_expect_refusal "$limited_rc" "$WORK/limited.log" "$WORK/limited.elf" "IR name-id pool overflowed" \
  || fail "limit=8: not a clean refusal: $(sounio_refusal_reason), rc=$limited_rc; log: $(tail -n 5 "$WORK/limited.log" | tr '\n' ' ')"
grep -q "refusing" "$WORK/limited.log" \
  || fail "limit=8: refused without saying it refuses to emit; log: $(tail -n 5 "$WORK/limited.log" | tr '\n' ' ')"

# (b) no limit: same program builds and runs.
( ulimit -s 1572864 2>/dev/null || true
  env -u SOUNIO_IR_NAME_POOL_LIMIT "$MADAROS" build "$WORK/prog.sio" "$WORK/ok.elf" ) >"$WORK/ok.log" 2>&1 \
  || { tail -n 20 "$WORK/ok.log" >&2; fail "unlimited build of the same program failed"; }
chmod +x "$WORK/ok.elf"
set +e
ok_out="$("$WORK/ok.elf" 2>&1)"; ok_rc=$?
set -e
[[ "$ok_rc" -eq 0 && "$ok_out" == "NAME_POOL_OK 42" ]] \
  || fail "unlimited program: rc=$ok_rc output=$(tr '\n' ' ' <<<"$ok_out")"

echo "[madaros-name-pool-overflow] PASS: SOUNIO_IR_NAME_POOL_LIMIT=8 refuses (rc=$limited_rc, 'IR name-id pool overflowed', no output file); the same program builds and prints NAME_POOL_OK 42 without it"
