#!/usr/bin/env bash
# Prove that the opt-in compact imported-IR oracle refuses unknown semantics.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND="$ROOT_DIR/self-hosted/compiler/module_frontend.sio"
DRIVER="$ROOT_DIR/self-hosted/compiler/module_native_driver.sio"
POSITIVE_DIR="$ROOT_DIR/tests/compiler/default_path_fidelity_gate"
NEGATIVE_DIR="$ROOT_DIR/tests/compiler/compact_imported_ir_fail_closed"
RAW_MADAROS="${SOUNIO_COMPACT_IMPORTED_IR_RAW_BIN:-}"
EXPECTED_RAW_SHA256="${SOUNIO_COMPACT_IMPORTED_IR_EXPECTED_RAW_SHA256:-}"
TIMEOUT_SECONDS="${SOUNIO_COMPACT_IMPORTED_IR_TIMEOUT_SECONDS:-180}"
SOURCE_ONLY=0

fail() {
  printf 'COMPACT_IMPORTED_IR_FAIL_CLOSED_FAIL reason=%s\n' "$1" >&2
  exit 1
}

if [[ "${1:-}" == "--source-only" ]]; then
  SOURCE_ONLY=1
elif [[ $# -ne 0 ]]; then
  fail unexpected_argument
fi

[[ -f "$FRONTEND" ]] || fail frontend_missing
[[ -f "$DRIVER" ]] || fail driver_missing
[[ -f "$POSITIVE_DIR/main.sio" ]] || fail positive_main_missing
[[ -f "$NEGATIVE_DIR/main.sio" ]] || fail negative_main_missing
[[ -f "$NEGATIVE_DIR/leaf.sio" ]] || fail negative_leaf_missing

python3 - "$FRONTEND" "$DRIVER" "$NEGATIVE_DIR/leaf.sio" <<'PY' || exit 1
import re
import sys
from pathlib import Path

frontend = Path(sys.argv[1]).read_text(encoding="utf-8")
driver = Path(sys.argv[2]).read_text(encoding="utf-8")
negative_leaf = Path(sys.argv[3]).read_text(encoding="utf-8")


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
    assert "var MODULE_FRONTEND_IMPORTED_SIMPLE_COMPLETE: bool = false" in frontend
    assert "var MODULE_FRONTEND_IMPORTED_SIMPLE_FAILURE_KIND: i64 = 0" in frontend

    reset = function_body(frontend, "module_frontend_imported_simple_global_reset")
    assert "MODULE_FRONTEND_IMPORTED_SIMPLE_COMPLETE = true" in reset
    assert "MODULE_FRONTEND_IMPORTED_SIMPLE_FAILURE_KIND = 0" in reset

    push = function_body(frontend, "module_frontend_imported_simple_global_push_fn")
    assert "MODULE_FRONTEND_IMPORTED_SIMPLE_FN_KINDS[idx as usize] = 0" in push
    assert "module_frontend_imported_simple_mark_failure(2, path)" in push
    print_start = push.index("if module_frontend_name_is_println(callee) ||")
    print_end = push.index("} else if callee.len > 0", print_start)
    print_block = push[print_start:print_end]
    assert "module_frontend_try_eval_source_expr_i64(" in print_block
    assert "module_frontend_source_first_int_after" not in print_block
    assert "if printed.0 {" in print_block
    assert "printed_ok = true" in print_block
    assert re.search(
        r"if printed_ok\s*\{\s*MODULE_FRONTEND_IMPORTED_SIMPLE_FN_KINDS\[idx as usize\] = 3",
        print_block,
    )
    assert re.search(
        r"if answer_value\.0\s*\{\s*MODULE_FRONTEND_IMPORTED_SIMPLE_FN_KINDS\[idx as usize\] = 3",
        print_block,
    )
    assert print_block.count("MODULE_FRONTEND_IMPORTED_SIMPLE_FN_KINDS[idx as usize] = 3") == 2

    lower_file = function_body(frontend, "module_frontend_lower_source_file_simple_global")
    assert "while i + 2 < size {" in lower_file
    assert "while i + 2 < size && MODULE_FRONTEND_IMPORTED_SIMPLE_FN_COUNT < 64" not in lower_file
    assert "module_frontend_imported_simple_mark_failure(2, path)" in lower_file
    assert "MODULE_FRONTEND_IMPORTED_SIMPLE_FN_KINDS[fn_index as usize] == 0" in lower_file
    assert "module_frontend_imported_simple_mark_failure(1, path)" in lower_file

    recursive = function_body(frontend, "module_frontend_lower_imported_simple_global_recursive")
    depth_guard = recursive.index("if depth > 16")
    assert recursive.index("module_frontend_imported_simple_mark_failure(3, path)", depth_guard) > depth_guard
    assert recursive.index("return false", depth_guard) > depth_guard

    loader = function_body(frontend, "load_multimodule_imported_simple_ir_global")
    assert "if !MODULE_FRONTEND_IMPORTED_SIMPLE_COMPLETE" in loader
    assert "MODULE_FRONTEND_IMPORTED_SIMPLE_FAILURE_KIND" in loader

    evaluator = function_body(frontend, "module_frontend_try_eval_source_body_i64")
    assert "module_frontend_source_is_match" not in evaluator
    assert "match tag" not in push

    compile_advanced = function_body(driver, "compile_multimodule_native_advanced")
    compact_start = compile_advanced.index('if str_eq(read_env("SOUNIO_ENABLE_COMPACT_IMPORTED_IR"), "1")')
    compact_block = compile_advanced[compact_start:]
    assert "return 1" in compact_block
    assert "return simple_rc" in compact_block
    assert "falling back to full IR path" not in compact_block

    assert "match tag" in negative_leaf
    assert "1 => 42" in negative_leaf and "_ => 7" in negative_leaf
except (AssertionError, ValueError) as error:
    print(f"COMPACT_IMPORTED_IR_FAIL_CLOSED_FAIL reason=source_contract_{error}", file=sys.stderr)
    raise SystemExit(1)
PY

if [[ "$SOURCE_ONLY" -eq 1 ]]; then
  printf '%s\n' 'COMPACT_IMPORTED_IR_FAIL_CLOSED_SOURCE_RECEIPT status=pass default_kind=unsupported completeness=explicit capacity=fail_closed depth=fail_closed println=resolved_only negative_fixture=match default_full_ir=untouched'
  exit 0
fi

[[ "$(uname -s 2>/dev/null || true)" == Linux ]] || fail linux_required
case "$(uname -m 2>/dev/null || true)" in
  x86_64|amd64) ;;
  *) fail x86_64_required ;;
esac
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail timeout_invalid
[[ -n "$RAW_MADAROS" ]] || fail raw_binary_not_set
[[ -n "$EXPECTED_RAW_SHA256" ]] || fail expected_raw_sha256_not_set
[[ -x "$RAW_MADAROS" ]] || fail raw_binary_not_executable
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
[[ "$(od -An -tx1 -N4 "$RAW_MADAROS" | tr -d ' \n')" == 7f454c46 ]] || fail raw_binary_not_elf
RAW_SHA256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$RAW_SHA256" == "$EXPECTED_RAW_SHA256" ]] || fail raw_sha256_mismatch

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-compact-fail-closed.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

run_compact_compile() {
  local source="$1"
  local elf="$2"
  local log="$3"
  set +e
  SOUNIO_ENABLE_COMPACT_IMPORTED_IR=1 \
    timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
    "$RAW_MADAROS" "$source" -o "$elf" >"$log" 2>&1
  COMPILE_RC=$?
  set -e
}

positive_count=0
for case_spec in 'greet_42.sio|42' 'greet_999.sio|999' 'greet_7.sio|7' 'greet_neg7.sio|-7'; do
  variant="${case_spec%%|*}"
  expected="${case_spec#*|}"
  label="${variant%.sio}"
  case_dir="$WORK/$label"
  mkdir -p "$case_dir"
  cp "$POSITIVE_DIR/main.sio" "$case_dir/main.sio"
  cp "$POSITIVE_DIR/$variant" "$case_dir/greet.sio"
  elf="$case_dir/program.elf"
  log="$case_dir/compile.log"
  stdout="$case_dir/stdout"
  expected_stdout="$case_dir/expected"

  run_compact_compile "$case_dir/main.sio" "$elf" "$log"
  [[ "$COMPILE_RC" -eq 0 ]] || fail "${label}_compile_rc_${COMPILE_RC}"
  grep -Fq 'compact modular IR table path' "$log" || fail "${label}_compact_marker_missing"
  ! grep -Fq 'COMPACT_IMPORTED_IR_REFUSAL' "$log" || fail "${label}_unexpected_refusal"
  [[ -s "$elf" ]] || fail "${label}_elf_missing"
  chmod +x "$elf"
  set +e
  timeout 30 "$elf" >"$stdout" 2>&1
  runtime_rc=$?
  set -e
  [[ "$runtime_rc" -eq 0 ]] || fail "${label}_runtime_rc_${runtime_rc}"
  printf '%s\n' "$expected" >"$expected_stdout"
  cmp -s "$expected_stdout" "$stdout" || fail "${label}_stdout_mismatch"
  positive_count=$((positive_count + 1))
done

negative_case="$WORK/negative"
mkdir -p "$negative_case"
cp "$NEGATIVE_DIR/main.sio" "$negative_case/main.sio"
cp "$NEGATIVE_DIR/leaf.sio" "$negative_case/leaf.sio"
negative_elf="$negative_case/program.elf"
negative_log="$negative_case/compile.log"

run_compact_compile "$negative_case/main.sio" "$negative_elf" "$negative_log"
[[ "$COMPILE_RC" -ne 0 ]] || fail negative_compile_false_green
grep -Eq '^COMPACT_IMPORTED_IR_REFUSAL reason=unsupported_function_shape path=.+/leaf\.sio$' "$negative_log" \
  || fail negative_causal_marker_missing
grep -Fq 'module_native_driver: compact IR load failed: unsupported_function_shape' "$negative_log" \
  || fail negative_driver_refusal_missing
[[ ! -e "$negative_elf" ]] || fail negative_elf_present
! grep -Fq 'falling back to full IR path' "$negative_log" || fail negative_fallback_present
! grep -Fq 'canonical AST closure full IR path' "$negative_log" || fail negative_canonical_path_entered

printf 'COMPACT_IMPORTED_IR_FAIL_CLOSED_RECEIPT status=pass raw_sha256=%s positive_literals=%s/4 unknown_shape_rc=nonzero causal_marker=unsupported_function_shape elf=absent fallback=none default_full_ir=untouched\n' \
  "$RAW_SHA256" "$positive_count"
