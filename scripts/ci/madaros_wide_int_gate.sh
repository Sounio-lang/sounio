#!/usr/bin/env bash
# Wide-integer gate: exercise i128/i256 type identity, type safety,
# explicit casts, source compile/run, and the IrWideAdd carry-chain witness.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/madaros}"
WORK="${SOUNIO_WIDE_INT_GATE_DIR:-$(mktemp -d /tmp/sounio-wide-int.XXXXXX)}"
KEEP_WORK="${SOUNIO_WIDE_INT_GATE_KEEP:-0}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

fail() {
  echo "[wide-int] FAIL: $*" >&2
  exit 1
}

pass() {
  echo "[wide-int] PASS: $*"
}

expect_exit() {
  local expected="$1"
  shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ "$rc" != "$expected" ]]; then
    fail "expected exit $expected, got $rc: $*"
  fi
}

expect_log_contains() {
  local needle="$1"
  local file="$2"
  if ! grep -Fq "$needle" "$file"; then
    echo "[wide-int] log tail for $file:" >&2
    tail -n 40 "$file" >&2 || true
    fail "missing log marker: $needle"
  fi
}

mkdir -p "$WORK"
printf '[wide-int] madaros=%s\n' "$MADAROS"
printf '[wide-int] work=%s\n' "$WORK"

# ---------------------------------------------------------------------------
# 1. Type identity: i128, i256, u128, u256 all pass check
# ---------------------------------------------------------------------------
printf 'fn main() -> i64 { let x: i128 = 1 as i128; 0 }\n' > "$WORK/i128_basic.sio"
printf 'fn main() -> i64 { let x: i256 = 1 as i256; 0 }\n' > "$WORK/i256_basic.sio"
printf 'fn main() -> i64 { let x: u128 = 1 as u128; 0 }\n' > "$WORK/u128_basic.sio"
printf 'fn main() -> i64 { let x: u256 = 1 as u256; 0 }\n' > "$WORK/u256_basic.sio"

for src in "$WORK"/i128_basic.sio "$WORK"/i256_basic.sio "$WORK"/u128_basic.sio "$WORK"/u256_basic.sio; do
  "$MADAROS" check "$src" >"$WORK/check_$(basename "$src" .sio).log" 2>&1
  expect_log_contains "check: OK" "$WORK/check_$(basename "$src" .sio).log"
done
pass "i128/i256/u128/u256 type identity"

# ---------------------------------------------------------------------------
# 2. Type safety: i128 != i256 (must FAIL check)
# ---------------------------------------------------------------------------
printf 'fn main() -> i64 { let x: i128 = 1 as i256; 0 }\n' > "$WORK/mismatch_128_256.sio"
expect_exit 1 "$MADAROS" check "$WORK/mismatch_128_256.sio" >"$WORK/mismatch_128_256.log" 2>&1
pass "i128 != i256 rejected by checker"

# ---------------------------------------------------------------------------
# 3. Type safety: u128 != i128 (must FAIL check)
# ---------------------------------------------------------------------------
printf 'fn main() -> i64 { let x: u128 = 1 as i128; 0 }\n' > "$WORK/mismatch_u128_i128.sio"
expect_exit 1 "$MADAROS" check "$WORK/mismatch_u128_i128.sio" >"$WORK/mismatch_u128_i128.log" 2>&1
pass "u128 != i128 rejected by checker"

# ---------------------------------------------------------------------------
# 4. Explicit cast: 1 as i128 and 1 as i256 pass check
# ---------------------------------------------------------------------------
printf 'fn main() -> i64 { let a: i128 = 42 as i128; let b: i256 = 99 as i256; 0 }\n' > "$WORK/cast.sio"
"$MADAROS" check "$WORK/cast.sio" >"$WORK/cast.log" 2>&1
expect_log_contains "check: OK" "$WORK/cast.log"
pass "explicit cast (1 as i128, 1 as i256)"

# ---------------------------------------------------------------------------
# 5. Function with i128 param/return: check + build + run
# ---------------------------------------------------------------------------
printf 'fn id(x: i128) -> i128 { x }\nfn main() -> i64 { let r = id(42 as i128); 0 }\n' > "$WORK/i128_fn.sio"
"$MADAROS" check "$WORK/i128_fn.sio" >"$WORK/i128_fn_check.log" 2>&1
expect_log_contains "check: OK" "$WORK/i128_fn_check.log"
"$MADAROS" build "$WORK/i128_fn.sio" -o "$WORK/i128_fn.elf" >"$WORK/i128_fn_build.log" 2>&1
[[ -s "$WORK/i128_fn.elf" ]] || fail "i128 function did not produce ELF"
chmod +x "$WORK/i128_fn.elf"
expect_exit 0 "$WORK/i128_fn.elf"
pass "i128 function check/build/run"

# ---------------------------------------------------------------------------
# 6. Function with i256 param/return: check + build + run
# ---------------------------------------------------------------------------
printf 'fn id(x: i256) -> i256 { x }\nfn main() -> i64 { let r = id(7 as i256); 0 }\n' > "$WORK/i256_fn.sio"
"$MADAROS" check "$WORK/i256_fn.sio" >"$WORK/i256_fn_check.log" 2>&1
expect_log_contains "check: OK" "$WORK/i256_fn_check.log"
"$MADAROS" build "$WORK/i256_fn.sio" -o "$WORK/i256_fn.elf" >"$WORK/i256_fn_build.log" 2>&1
[[ -s "$WORK/i256_fn.elf" ]] || fail "i256 function did not produce ELF"
chmod +x "$WORK/i256_fn.elf"
expect_exit 0 "$WORK/i256_fn.elf"
pass "i256 function check/build/run"

# ---------------------------------------------------------------------------
# 7. Wide-add4 witness: 4-limb carry chain proves real i256 arithmetic.
#    (-1,-1,-1,0) + (1,0,0,0) -> carry cascades 0->1->2->3; high limb = 1.
#    Fake-i64 would give 0. Exit code MUST be 1.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-add4 "$WORK/wide_add4.elf" >"$WORK/wide_add4.log" 2>&1
expect_log_contains "wide-add4 rc=0" "$WORK/wide_add4.log"
[[ -s "$WORK/wide_add4.elf" ]] || fail "wide-add4 did not produce ELF"
file "$WORK/wide_add4.elf" | grep -Fq "ELF 64-bit" || fail "wide-add4 not ELF64"
chmod +x "$WORK/wide_add4.elf"
expect_exit 1 "$WORK/wide_add4.elf"
pass "IrWideAdd 4-limb carry chain (exit=1, not fake-i64 exit=0)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "[wide-int] PASS: type identity, type safety, casts, i128/i256 compile, IrWideAdd witness"
echo "[wide-int] NOTE: IrWideSub/Mul/Div/Mod/Shr/Cmp not yet codegen-supported"
echo "[wide-int] NOTE: source-level wide arithmetic lowering not yet implemented"
