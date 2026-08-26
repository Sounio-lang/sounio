#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOURCE="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
SEED="$ROOT_DIR/bin/souc-lean-single-x86_64"
POSITIVE="$ROOT_DIR/tests/run-pass/borrow_local_array_element_ref.sio"
MUTABILITY="$ROOT_DIR/tests/compile-fail/borrow_mut_immutable_local_array_element.sio"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-issue1678-local-borrow.XXXXXX")"

fail() {
  printf 'lean-single-local-array-element-borrow: FAIL: %s test_root=%s\n' "$*" "$TEST_ROOT" >&2
  exit 1
}

cleanup() {
  if [[ "${SOUNIO_KEEP_ISSUE1678_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

for command in cmp llvm-objdump perl readelf sha256sum; do
  command -v "$command" >/dev/null 2>&1 || fail "required command is missing: $command"
done
[[ -x "$SEED" ]] || fail "canonical lean_single seed is missing: $SEED"

rebuilt="$TEST_ROOT/souc-rebuilt"
"$SEED" "$SOURCE" "$rebuilt" --target x86_64-linux \
  >"$TEST_ROOT/rebuild.log" 2>&1 || fail "canonical seed could not rebuild lean_single"
chmod +x "$rebuilt"
cmp -s "$SEED" "$rebuilt" || fail "canonical seed is not the source fixed point"
fixed_sha="$(sha256sum "$SEED" | awk '{print $1}')"

positive_x86="$TEST_ROOT/positive-x86"
"$SEED" "$POSITIVE" "$positive_x86" \
  >"$TEST_ROOT/positive-x86.log" 2>&1 || fail "x86 positive witness did not compile"
chmod +x "$positive_x86"
"$positive_x86" || fail "x86 positive witness did not execute successfully"

if "$SEED" "$MUTABILITY" "$TEST_ROOT/mutability" \
    >"$TEST_ROOT/mutability.log" 2>&1; then
  fail "exclusive element borrow accepted an immutable array"
fi
grep -Fq 'Mut borrow requires mutable array or slice source' \
  "$TEST_ROOT/mutability.log" || fail "immutable-array refusal used the wrong rule"

cat >"$TEST_ROOT/oob.sio" <<'SOUNIO'
fn read_i64(p: &i64) -> i64 with Mut, Panic { (*p) }
fn main() -> i64 with Mut, Panic {
    let values = [10, 20, 30, 40]
    read_i64(&values[4 as usize])
}
SOUNIO
"$SEED" "$TEST_ROOT/oob.sio" "$TEST_ROOT/oob" \
  >"$TEST_ROOT/oob-compile.log" 2>&1 || fail "out-of-bounds witness did not compile"
chmod +x "$TEST_ROOT/oob"
set +e
"$TEST_ROOT/oob" >"$TEST_ROOT/oob-run.log" 2>&1
oob_rc=$?
set -e
[[ "$oob_rc" -eq 1 ]] || fail "out-of-bounds element borrow did not refuse with rc=1 (rc=$oob_rc)"

positive_a64="$TEST_ROOT/positive-aarch64"
"$SEED" "$POSITIVE" "$positive_a64" --target aarch64-linux \
  >"$TEST_ROOT/positive-aarch64.log" 2>&1 || fail "AArch64 positive witness did not compile"
readelf -h "$positive_a64" | grep -Fq 'Machine:                           AArch64' || \
  fail "AArch64 target emitted the wrong ELF machine"
llvm-objdump -d --triple=aarch64-linux-gnu "$positive_a64" \
  >"$TEST_ROOT/positive-aarch64.dis"
grep -Eq 'cmp[[:space:]]+x9, x10' "$TEST_ROOT/positive-aarch64.dis" || \
  fail "AArch64 output omitted the index bound comparison"
grep -Eq 'ldr[[:space:]]+x0, \[x0, x9, lsl #3\]' "$TEST_ROOT/positive-aarch64.dis" || \
  fail "AArch64 output omitted aggregate pointer-slot loading"
grep -Eq 'add[[:space:]]+x0, x0, x9, lsl #3' "$TEST_ROOT/positive-aarch64.dis" || \
  fail "AArch64 output omitted scalar cell addressing"

make_sabotage() {
  local marker="$1" output="$2"
  cp "$SOURCE" "$output"
  perl -0pi -e "s/($marker\\n[[:space:]]*)if TK\\[EP as usize\\] == 42 \\{/\${1}if false {/" \
    "$output"
  grep -A1 "$marker" "$output" | grep -Fq 'if false {' || \
    fail "sabotage did not disable $marker"
}

x86_sabotage_source="$TEST_ROOT/lean-single-x86-sabotage.sio"
make_sabotage ISSUE1678_SINGLE_ELEMENT_BORROW_X86 "$x86_sabotage_source"
x86_sabotage_compiler="$TEST_ROOT/souc-x86-sabotage"
"$SEED" "$x86_sabotage_source" "$x86_sabotage_compiler" --target x86_64-linux \
  >"$TEST_ROOT/x86-sabotage-build.log" 2>&1 || fail "x86 sabotage compiler did not build"
chmod +x "$x86_sabotage_compiler"
if "$x86_sabotage_compiler" "$POSITIVE" "$TEST_ROOT/x86-sabotage-output" \
    >"$TEST_ROOT/x86-sabotage.log" 2>&1; then
  fail "x86 same-source sabotage still admitted the positive witness"
fi
grep -Fq 'error[E001]: Type mismatch in call argument' "$TEST_ROOT/x86-sabotage.log" || \
  fail "x86 sabotage refused through an incidental rule"

a64_sabotage_source="$TEST_ROOT/lean-single-a64-sabotage.sio"
make_sabotage ISSUE1678_SINGLE_ELEMENT_BORROW_A64 "$a64_sabotage_source"
a64_sabotage_compiler="$TEST_ROOT/souc-a64-sabotage"
"$SEED" "$a64_sabotage_source" "$a64_sabotage_compiler" --target x86_64-linux \
  >"$TEST_ROOT/a64-sabotage-build.log" 2>&1 || fail "AArch64 sabotage compiler did not build"
chmod +x "$a64_sabotage_compiler"
"$a64_sabotage_compiler" "$POSITIVE" "$TEST_ROOT/a64-sabotage-output" \
  --target aarch64-linux >"$TEST_ROOT/a64-sabotage.log" 2>&1 || true
grep -Fq 'error[E001]: Type mismatch in call argument' "$TEST_ROOT/a64-sabotage.log" || \
  fail "AArch64 sabotage refused through an incidental rule"

printf 'lean-single-local-array-element-borrow: PASS fixed_point_sha256=%s x86=EXECUTED aarch64=CODEGEN_VERIFIED aggregate_slot=LOADED scalar_cell=ADDRESSED bounds=REFUSED mutability=REFUSED slice=PRESERVED slice_element=EXECUTED sabotage_x86=E001 sabotage_a64=E001\n' \
  "$fixed_sha"
