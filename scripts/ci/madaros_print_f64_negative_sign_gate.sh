#!/usr/bin/env bash
# Regression gate for #890: the '-' write syscall must not destroy f64 magnitude bits.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODEGEN="$ROOT_DIR/self-hosted/native/codegen_x86_linux.sio"
SOURCE="$ROOT_DIR/tests/regression/print_f64_negative_sign.sio"
EXPECTED="$ROOT_DIR/tests/regression/print_f64_negative_sign.stdout"
WRAPPER="$ROOT_DIR/bin/madaros"
RAW_MADAROS="${SOUNIO_MADAROS_PRINT_F64_RAW_BIN:-}"
EXPECTED_RAW_SHA256="${SOUNIO_MADAROS_PRINT_F64_EXPECTED_SHA256:-}"
KEEP_WORK="${SOUNIO_MADAROS_PRINT_F64_KEEP:-0}"
TIMEOUT_SECONDS="${SOUNIO_MADAROS_PRINT_F64_TIMEOUT_SECONDS:-360}"
RUNTIME_TIMEOUT_SECONDS="${SOUNIO_MADAROS_PRINT_F64_RUNTIME_TIMEOUT_SECONDS:-30}"
MODE="source-only"

fail() {
  printf 'MADAROS_PRINT_F64_NEGATIVE_SIGN_FAIL reason=%s\n' "$1" >&2
  exit 1
}

usage() {
  cat <<'EOF'
usage: scripts/ci/madaros_print_f64_negative_sign_gate.sh [--source-only|--source-fresh]

  --source-only       Verify backend source shape and the pinned regression fixture.
  --source-fresh      Require an explicit raw Madaros ELF and exact SHA-256.

Source-fresh mode requires:
  SOUNIO_MADAROS_PRINT_F64_RAW_BIN=/path/to/madaros
  SOUNIO_MADAROS_PRINT_F64_EXPECTED_SHA256=<64 lowercase hex>
EOF
}

case "${1:-}" in
  ""|--source-only) MODE="source-only" ;;
  --source-fresh) MODE="source-fresh" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; fail unexpected_argument ;;
esac
[[ $# -le 1 ]] || fail unexpected_argument

for path in "$CODEGEN" "$SOURCE" "$EXPECTED" "$WRAPPER"; do
  [[ -f "$path" ]] || fail "missing_${path#"$ROOT_DIR"/}"
done
[[ -x "$WRAPPER" ]] || fail wrapper_not_executable

print_body="$(sed -n '/^fn emit_builtin_print_f64(/,/^fn emit_builtin_str_len(/p' "$CODEGEN")"
[[ "$(grep -Fc 'preserve bits across sign write syscall' <<<"$print_body")" -eq 1 ]] ||
  fail preserved_bits_store_count
grep -Fq 'c.code = emit_store_rax_rbp_disp32(c.code, -48)   // preserve bits across sign write syscall' <<<"$print_body" ||
  fail preserved_bits_store_missing
grep -Fq 'c.code = emit_load_rbp_disp32_rax(c.code, -48)' <<<"$print_body" ||
  fail preserved_bits_reload_missing

negative_path="$(sed -n '/\/\/ Negative path: emit '\''-'\''/,/\/\/ Patch jns displacement/p' "$CODEGEN")"
grep -Fq 'c = emit_write_syscall_for_target(c, c.target_os_id)' <<<"$negative_path" ||
  fail sign_write_syscall_missing
grep -Fq 'c.code = emit_load_rbp_disp32_rax(c.code, -48)' <<<"$negative_path" ||
  fail post_syscall_bits_reload_missing
if grep -Fq 'c.code = emit_mov_reg_reg(c.code, 0, 7)           // mov rax, rdi' <<<"$negative_path"; then
  fail post_syscall_destroyed_rdi_reuse_present
fi

for anchor in \
  'println(keep_f64(0.0 - 0.2))' \
  'println(keep_f64(0.0 - 1.5))' \
  'println(keep_f64(0.2))' \
  'print_int(keep_i64(42))'; do
  grep -Fq "$anchor" "$SOURCE" || fail "fixture_missing_${anchor//[^a-zA-Z0-9]/_}"
done
[[ "$(cat "$EXPECTED")" == $'-0.200000\n-1.500000\n0.200000\n42\nPRINT_F64_NEGATIVE_SIGN_PASS' ]] ||
  fail expected_stdout_contract

printf 'MADAROS_PRINT_F64_NEGATIVE_SIGN_SOURCE_PASS issue=890 sign_bits=stack_preserved sign_syscall=clobber_safe cases=negative_fraction,negative_mixed,positive control=print_int runtime=not_run\n'

if [[ "$MODE" == source-only ]]; then
  exit 0
fi

if [[ -n "${SOUNIO_MADAROS_PRINT_F64_WORK_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_PRINT_F64_WORK_DIR"
  [[ ! -e "$WORK" ]] || fail work_directory_exists
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-print-f64-negative.XXXXXX")"
fi
if [[ "$KEEP_WORK" != 1 ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

[[ -n "$RAW_MADAROS" ]] || fail explicit_source_fresh_compiler_required
[[ -x "$RAW_MADAROS" ]] || fail source_fresh_compiler_not_executable
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd -P)/$(basename "$RAW_MADAROS")"
[[ "$(od -An -tx1 -N4 "$RAW_MADAROS" | tr -d ' \n')" == 7f454c46 ]] ||
  fail source_fresh_compiler_must_be_elf
[[ "$EXPECTED_RAW_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail expected_compiler_sha256_required
compiler_sha256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$compiler_sha256" == "$EXPECTED_RAW_SHA256" ]] || fail source_fresh_compiler_sha256_mismatch
RUNNER=(env \
  -u SOUNIO_MADAROS_BIN \
  -u SOUNIO_SOUC_BIN \
  "MADAROS_RAW_BIN=$RAW_MADAROS" \
  "SOUNIO_STDLIB_PATH=$ROOT_DIR/stdlib" \
  SOUNIO_SOUC_ENGINE=madaros \
  SOUNIO_ENABLE_COMPACT_IMPORTED_IR=0 \
  OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=0 \
  "$WRAPPER")
PROVENANCE=source-fresh
COMPILER_SURFACE="$RAW_MADAROS"

ELF="$WORK/print-f64-negative.elf"
CHECK_LOG="$WORK/check.log"
COMPILE_LOG="$WORK/compile.log"
ACTUAL="$WORK/runtime.stdout"
STDERR="$WORK/runtime.stderr"

if ! timeout --signal=TERM --kill-after=5s "$TIMEOUT_SECONDS" \
    "${RUNNER[@]}" check "$SOURCE" >"$CHECK_LOG" 2>&1; then
  tail -n 120 "$CHECK_LOG" >&2 || true
  fail check_failed
fi
if grep -Eiq 'segmentation fault|core dumped|fatal:|error\[E|^error:' "$CHECK_LOG"; then
  tail -n 120 "$CHECK_LOG" >&2 || true
  fail check_diagnostic_or_crash
fi

rm -f "$ELF"
if ! timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
    "${RUNNER[@]}" --science-boundary off build "$SOURCE" -o "$ELF" >"$COMPILE_LOG" 2>&1; then
  tail -n 120 "$COMPILE_LOG" >&2 || true
  fail compile_failed
fi
if grep -Eq 'native_prebundle:|falling back to full IR path|compact modular IR table path|legacy compact IR differential enabled|SELFHOST=fallback|driver_orchestration.*status=fallback' "$COMPILE_LOG"; then
  tail -n 120 "$COMPILE_LOG" >&2 || true
  fail compact_or_fallback_path_observed
fi
if grep -Eiq 'segmentation fault|core dumped|fatal:|error\[E|^error:' "$COMPILE_LOG"; then
  tail -n 120 "$COMPILE_LOG" >&2 || true
  fail compile_diagnostic_or_crash
fi
[[ -s "$ELF" && -x "$ELF" ]] || fail final_elf_missing_or_not_executable
[[ "$(od -An -tx1 -N4 "$ELF" | tr -d ' \n')" == 7f454c46 ]] || fail final_output_not_elf

set +e
timeout --signal=TERM --kill-after=5s "$RUNTIME_TIMEOUT_SECONDS" \
  "$ELF" >"$ACTUAL" 2>"$STDERR"
runtime_rc=$?
set -e
[[ "$runtime_rc" -eq 0 ]] || {
  cat "$ACTUAL" >&2 || true
  cat "$STDERR" >&2 || true
  fail "runtime_rc_${runtime_rc}"
}
[[ ! -s "$STDERR" ]] || fail runtime_stderr_not_empty
if ! cmp -s "$EXPECTED" "$ACTUAL"; then
  diff -u "$EXPECTED" "$ACTUAL" >&2 || true
  fail runtime_stdout_mismatch
fi

elf_sha256="$(sha256sum "$ELF" | awk '{print $1}')"
printf 'MADAROS_PRINT_F64_NEGATIVE_SIGN_PASS issue=890 compiler_provenance=%s compiler_surface=%s compiler_sha256=%s driver=full_ir_noncompact elf_sha256=%s runtime_exit=0 stdout=negative_0.2,negative_1.5,positive_0.2,print_int_42 fallback=none\n' \
  "$PROVENANCE" "$COMPILER_SURFACE" "$compiler_sha256" "$elf_sha256"
