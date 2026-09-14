#!/usr/bin/env bash
# KL-7 wide-integer safety: fail-closed wide print + signed high-limb witnesses.
# Legacy emit-wide-* / full madaros_wide_int_gate.sh stays separate (arena IR).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/madaros}"
WORK="${SOUNIO_KL7_WIDE_PRINT_GATE_DIR:-$(mktemp -d /tmp/sounio-kl7-wide.XXXXXX)}"
KEEP_WORK="${SOUNIO_KL7_WIDE_PRINT_GATE_KEEP:-0}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

fail() {
  echo "[kl7-wide] FAIL: $*" >&2
  exit 1
}

pass() {
  echo "[kl7-wide] PASS: $*"
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
    echo "[kl7-wide] log tail for $file:" >&2
    tail -n 40 "$file" >&2 || true
    fail "missing log marker: $needle"
  fi
}

mkdir -p "$WORK"
printf '[kl7-wide] madaros=%s\n' "$MADAROS"
printf '[kl7-wide] work=%s\n' "$WORK"

# Fail-closed print: typecheck OK; build refuses; no ELF.
WIDE_PRINT_FIXTURE="$ROOT_DIR/tests/compile-fail/kl7_wide_int_print.sio"
"$MADAROS" check "$WIDE_PRINT_FIXTURE" >"$WORK/kl7_print_check.log" 2>&1
expect_log_contains "check: OK" "$WORK/kl7_print_check.log"
rm -f "$WORK/kl7_print.elf"
expect_exit 1 "$MADAROS" build "$WIDE_PRINT_FIXTURE" -o "$WORK/kl7_print.elf" >"$WORK/kl7_print_build.log" 2>&1
expect_log_contains "wide integer print is not implemented; use shifts and casts to print verified 64-bit limbs" "$WORK/kl7_print_build.log"
[[ ! -e "$WORK/kl7_print.elf" ]] || fail "wide println refusal still produced an ELF"
pass "wide println fails closed before ELF emission"

# Narrow control: i64 println(-1) still works.
cat >"$WORK/kl7_print_narrow_control.sio" <<'EOF'
fn main() -> i64 with IO {
    let x: i64 = 0 - 1
    println(x)
    0
}
EOF
"$MADAROS" check "$WORK/kl7_print_narrow_control.sio" >"$WORK/kl7_print_narrow_check.log" 2>&1
expect_log_contains "check: OK" "$WORK/kl7_print_narrow_check.log"
"$MADAROS" build "$WORK/kl7_print_narrow_control.sio" -o "$WORK/kl7_print_narrow.elf" >"$WORK/kl7_print_narrow_build.log" 2>&1
chmod +x "$WORK/kl7_print_narrow.elf"
"$WORK/kl7_print_narrow.elf" >"$WORK/kl7_print_narrow.stdout"
[[ "$(cat "$WORK/kl7_print_narrow.stdout")" == "-1" ]] || fail "narrow sabotage control did not print -1"
pass "print refusal narrow control runs and prints -1"

# Signed widening / unary negation witness.
SIGN_FIXTURE="$ROOT_DIR/tests/run-pass/kl7_wide_sign_extension.sio"
"$MADAROS" check "$SIGN_FIXTURE" >"$WORK/kl7_sign_check.log" 2>&1
expect_log_contains "check: OK" "$WORK/kl7_sign_check.log"
"$MADAROS" build "$SIGN_FIXTURE" -o "$WORK/kl7_sign.elf" >"$WORK/kl7_sign_build.log" 2>&1
chmod +x "$WORK/kl7_sign.elf"
"$WORK/kl7_sign.elf" >"$WORK/kl7_sign.stdout"
[[ "$(cat "$WORK/kl7_sign.stdout")" == "KL7_WIDE_SIGN_EXTENSION_OK" ]] || fail "signed-wide witness output/exit mismatch"
pass "i256/i512 sign extension, signed comparison and -(2^200)"

set +e
SOUNIO_WIDE_MUL_SABOTAGE=1 "$MADAROS" build "$SIGN_FIXTURE" -o "$WORK/kl7_sign_sabotaged.elf" >"$WORK/kl7_sign_sabotaged_build.log" 2>&1
sabotage_build_rc=$?
set -e
if [[ "$sabotage_build_rc" == "0" ]]; then
  chmod +x "$WORK/kl7_sign_sabotaged.elf"
  set +e
  "$WORK/kl7_sign_sabotaged.elf" >"$WORK/kl7_sign_sabotaged.stdout" 2>&1
  sabotage_run_rc=$?
  set -e
  [[ "$sabotage_run_rc" != "0" ]] || fail "wide-mul sabotage did not break KL-7 sign witness"
fi
pass "sign-extension sabotage control fires"

echo "MADAROS_KL7_WIDE_PRINT_GATE_OK"
