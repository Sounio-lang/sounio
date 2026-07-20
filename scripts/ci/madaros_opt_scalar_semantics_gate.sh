#!/usr/bin/env bash
# Issue #1070: keep optimized scalar calls semantically equal to native no-opt.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPT_CLEANUP="$ROOT_DIR/self-hosted/ir/opt_cleanup.sio"
FIXTURE_DIR="$ROOT_DIR/tests/compiler/madaros_opt_scalar_semantics"
COMPILER_CLI="$ROOT_DIR/bin/souc"
RAW_MADAROS="${SOUNIO_ISSUE_1070_RAW_BIN:-${MADAROS_RAW_BIN:-}}"
EXPECTED_RAW_SHA256="${SOUNIO_ISSUE_1070_EXPECTED_RAW_SHA256:-}"
OLD_RAW_SHA256="61f64740f83f004eaac319cdedebb9f8d3164cfe57056d169835273f067c7f86"
TIMEOUT_SECONDS="${SOUNIO_ISSUE_1070_TIMEOUT_SECONDS:-60}"
MODE="runtime"

fail() {
  printf 'MADAROS_OPT_SCALAR_SEMANTICS_FAIL mode=%s reason=%s\n' "$MODE" "$1" >&2
  exit 1
}

case "${1:-}" in
  "") ;;
  --source-only) MODE="source-only" ;;
  --old-raw-negative) MODE="old-raw-negative" ;;
  *) fail unexpected_argument ;;
esac
[[ $# -le 1 ]] || fail unexpected_argument

for path in \
  "$OPT_CLEANUP" \
  "$FIXTURE_DIR/addition.sio" \
  "$FIXTURE_DIR/bitwise_calls.sio" \
  "$FIXTURE_DIR/bitwise_calls.stdout" \
  "$FIXTURE_DIR/equality_guard.sio" \
  "$FIXTURE_DIR/equality_guard.stdout"; do
  [[ -f "$path" ]] || fail required_file_missing
done

python3 - "$OPT_CLEANUP" "$FIXTURE_DIR" <<'PY' || exit 1
import re
import sys
from pathlib import Path

optimizer_path = Path(sys.argv[1])
fixture_dir = Path(sys.argv[2])
optimizer = optimizer_path.read_text(encoding="utf-8")


def function_body(source: str, name: str) -> str:
    match = re.search(r"(?:pub\s+)?fn\s+" + re.escape(name) + r"\s*\(", source)
    if match is None:
        raise AssertionError(f"missing_function_{name}")
    start = source.find("{", match.end())
    if start < 0:
        raise AssertionError(f"missing_body_{name}")
    depth = 0
    for pos in range(start, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1]
    raise AssertionError(f"unterminated_function_{name}")


try:
    call_liveness = function_body(optimizer, "ocp_mark_call_args_used")
    assert "while count < 256" in call_liveness
    assert "ocp_mark_used_reg(used, (*node).head)" in call_liveness
    assert "cur = (*node).tail" in call_liveness

    mark_used = function_body(optimizer, "ocp_mark_used")
    assert "ocp_mark_call_args_used(&! used, instr.call_args)" in mark_used
    assert "instr.op == IrOpcode::IrIndexSet" in mark_used
    assert "ocp_mark_used_reg(&! used, instr.imm_i64)" in mark_used

    dse_args = function_body(optimizer, "ocp_dse_mark_call_args_read")
    assert "last_write[reg as usize] = -1" in dse_args
    assert "cur = (*node).tail" in dse_args

    dse = function_body(optimizer, "ocp_dse")
    assert "ocp_dse_mark_call_args_read(&! last_write, instr.call_args)" in dse
    assert "instr.op == IrOpcode::IrIndexSet" in dse
    assert "last_write[instr.imm_i64 as usize] = -1" in dse

    boxed = function_body(optimizer, "ocp_has_boxed_call_operands")
    assert "instr.arg_count > 0" in boxed
    assert "match instr.call_args" in boxed
    assert "Some(_) => { return true }" in boxed

    compact = function_body(optimizer, "ocp_compact_nops")
    guard = "if ocp_has_boxed_call_operands(&func) { return func }"
    assert guard in compact
    assert compact.index(guard) < compact.index("var result = func")

    pipeline = function_body(optimizer, "opt_cleanup_function_with_algebras_and_audit")
    assert "ocp_dedup_imm(func)" in pipeline
    assert "ocp_const_fold_with_audit" in pipeline
    assert "ocp_dce_once" in pipeline

    addition = (fixture_dir / "addition.sio").read_text(encoding="utf-8")
    bitwise = (fixture_dir / "bitwise_calls.sio").read_text(encoding="utf-8")
    equality = (fixture_dir / "equality_guard.sio").read_text(encoding="utf-8")
    assert "add(1) - 8" in addition
    assert "fn sentinel() -> i64" in addition
    assert "apply_xor_mask(877, 7)" in bitwise
    assert "(ab ^ 874) | (ba ^ 879) | (left ^ 5) | (right ^ 866)" in bitwise
    assert "if observed != 877" in equality
    assert (fixture_dir / "bitwise_calls.stdout").read_text(encoding="utf-8") == "874 879 5 866\n"
    assert (fixture_dir / "equality_guard.stdout").read_text(encoding="utf-8") == "877\n"
except (AssertionError, ValueError) as error:
    print(
        f"MADAROS_OPT_SCALAR_SEMANTICS_FAIL mode=source-contract reason={error}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

if [[ "$MODE" == "source-only" ]]; then
  printf '%s\n' 'MADAROS_OPT_SCALAR_SEMANTICS_SOURCE_PASS issue=1070 liveness=src1,src2,call_args,index_set dse=call_args,index_set nop_compaction=boxed_call_quarantined optimizer_pipeline=active runtime=not_run merge_ready=0'
  exit 0
fi

[[ "$(uname -s 2>/dev/null || true)" == Linux ]] || fail linux_required
case "$(uname -m 2>/dev/null || true)" in
  x86_64|amd64) ;;
  *) fail x86_64_required ;;
esac
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail timeout_invalid
[[ -x "$COMPILER_CLI" ]] || fail compiler_cli_not_executable
[[ -n "$RAW_MADAROS" ]] || fail raw_binary_not_set
[[ -x "$RAW_MADAROS" ]] || fail raw_binary_not_executable
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
[[ "$(od -An -tx1 -N4 "$RAW_MADAROS" | tr -d ' \n')" == 7f454c46 ]] || fail raw_binary_not_elf
RAW_SHA256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"

if [[ "$MODE" == "old-raw-negative" ]]; then
  [[ "$RAW_SHA256" == "$OLD_RAW_SHA256" ]] || fail old_raw_sha256_mismatch
else
  [[ "$EXPECTED_RAW_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail expected_raw_sha256_required
  [[ "$RAW_SHA256" == "$EXPECTED_RAW_SHA256" ]] || fail raw_sha256_mismatch
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-issue1070-opt.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

COMPILE_RC=0
COMPILE_ELF=""
COMPILE_LOG=""
RUNTIME_RC=0
RUNTIME_STDOUT=""
RUNTIME_STDERR=""

run_compile() {
  local label="$1"
  local variant="$2"
  local source="$FIXTURE_DIR/$label.sio"
  local -a mode_args=()
  if [[ "$variant" == opt ]]; then
    mode_args=(-O)
  else
    mode_args=(-t native)
  fi
  COMPILE_ELF="$WORK/$label.$variant.elf"
  COMPILE_LOG="$WORK/$label.$variant.compile.log"
  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" SOUNIO_SOUC_ENGINE=madaros \
    timeout --signal=TERM --kill-after=5s "$TIMEOUT_SECONDS" \
    "$COMPILER_CLI" "${mode_args[@]}" "$source" -o "$COMPILE_ELF" >"$COMPILE_LOG" 2>&1
  COMPILE_RC=$?
  set -e
}

run_elf() {
  local label="$1"
  local variant="$2"
  local elf="$3"
  chmod +x "$elf"
  RUNTIME_STDOUT="$WORK/$label.$variant.stdout"
  RUNTIME_STDERR="$WORK/$label.$variant.stderr"
  set +e
  timeout --signal=TERM --kill-after=5s "$TIMEOUT_SECONDS" \
    "$elf" >"$RUNTIME_STDOUT" 2>"$RUNTIME_STDERR"
  RUNTIME_RC=$?
  set -e
}

require_clean_compile_log() {
  local label="$1"
  grep -Eq '^Merged IR: [1-9][0-9]*$' "$COMPILE_LOG" || fail "${label}_merged_ir_missing"
  if grep -Eqi 'falling back|lean_single' "$COMPILE_LOG"; then
    fail "${label}_fallback_observed"
  fi
}

require_expected_stdout() {
  local label="$1"
  local stdout_path="$2"
  if [[ "$label" == addition ]]; then
    [[ ! -s "$stdout_path" ]] || fail "${label}_stdout_mismatch"
  else
    cmp -s "$FIXTURE_DIR/$label.stdout" "$stdout_path" || fail "${label}_stdout_mismatch"
  fi
}

verify_good_variant() {
  local label="$1"
  local variant="$2"
  run_compile "$label" "$variant"
  [[ "$COMPILE_RC" -eq 0 ]] || fail "${label}_${variant}_compile_rc_${COMPILE_RC}"
  require_clean_compile_log "${label}_${variant}"
  [[ -s "$COMPILE_ELF" ]] || fail "${label}_${variant}_elf_missing"
  run_elf "$label" "$variant" "$COMPILE_ELF"
  [[ "$RUNTIME_RC" -eq 0 ]] || fail "${label}_${variant}_runtime_rc_${RUNTIME_RC}"
  require_expected_stdout "$label" "$RUNTIME_STDOUT"
}

if [[ "$MODE" == "old-raw-negative" ]]; then
  for label in addition bitwise_calls equality_guard; do
    verify_good_variant "$label" noopt
  done

  run_compile addition opt
  [[ "$COMPILE_RC" -eq 139 ]] || fail "addition_opt_compile_rc_${COMPILE_RC}"
  require_clean_compile_log addition_opt
  grep -Fq 'Segmentation fault' "$COMPILE_LOG" || fail addition_opt_segfault_marker_missing
  [[ ! -s "$COMPILE_ELF" ]] || fail addition_opt_unexpected_elf

  run_compile bitwise_calls opt
  [[ "$COMPILE_RC" -eq 0 ]] || fail "bitwise_opt_compile_rc_${COMPILE_RC}"
  require_clean_compile_log bitwise_opt
  run_elf bitwise_calls opt "$COMPILE_ELF"
  [[ "$RUNTIME_RC" -eq 111 ]] || fail "bitwise_opt_runtime_rc_${RUNTIME_RC}"
  printf '874 0 0 8\n' >"$WORK/bitwise_calls.old_raw.stdout"
  cmp -s "$WORK/bitwise_calls.old_raw.stdout" "$RUNTIME_STDOUT" || fail bitwise_opt_stdout_changed

  run_compile equality_guard opt
  [[ "$COMPILE_RC" -eq 0 ]] || fail "equality_opt_compile_rc_${COMPILE_RC}"
  require_clean_compile_log equality_opt
  run_elf equality_guard opt "$COMPILE_ELF"
  [[ "$RUNTIME_RC" -eq 1 ]] || fail "equality_opt_runtime_rc_${RUNTIME_RC}"
  cmp -s "$FIXTURE_DIR/equality_guard.stdout" "$RUNTIME_STDOUT" || fail equality_opt_stdout_changed

  printf 'MADAROS_OPT_SCALAR_SEMANTICS_OLD_RAW_NEGATIVE_PASS issue=1070 raw_sha256=%s noopt_parity=3/3 addition_opt=compile_rc139 bitwise_opt=stdout_874_0_0_8,rc111 equality_opt=stdout_877,rc1 fallback=0 patched_runtime=not_run merge_ready=0\n' "$RAW_SHA256"
  exit 0
fi

for label in addition bitwise_calls equality_guard; do
  verify_good_variant "$label" noopt
  verify_good_variant "$label" opt
  cmp -s "$WORK/$label.noopt.stdout" "$WORK/$label.opt.stdout" || fail "${label}_runtime_parity_mismatch"
done

printf 'MADAROS_OPT_SCALAR_SEMANTICS_RUNTIME_PASS issue=1070 raw_sha256=%s optimized_parity=3/3 addition=rc0 bitwise=stdout_874_879_5_866,rc0 equality=stdout_877,rc0 fallback=0 merge_ready=1\n' "$RAW_SHA256"
