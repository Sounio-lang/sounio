#!/usr/bin/env bash
# Prove that the canonical native-v2 source path compiles one collected closure.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="$ROOT_DIR/self-hosted/compiler/main.sio"
FRONTEND="$ROOT_DIR/self-hosted/compiler/module_frontend.sio"
LAUNCHER="$ROOT_DIR/bin/madaros"
FIXTURE="$ROOT_DIR/tests/compiler/module_closure_single_compile/main.sio"
NO_IMPORT_FIXTURE="$ROOT_DIR/tests/madaros/source_to_elf/lit42_exit42.sio"
RAW_MADAROS="${SOUNIO_SINGLE_CLOSURE_RAW_BIN:-}"
EXPECTED_RAW_SHA256="${SOUNIO_SINGLE_CLOSURE_EXPECTED_RAW_SHA256:-}"
SOURCE_ONLY=0

fail() {
  printf 'MADAROS_SINGLE_CLOSURE_COMPILE_FAIL reason=%s\n' "$1" >&2
  exit 1
}

if [[ "${1:-}" == "--source-only" ]]; then
  SOURCE_ONLY=1
elif [[ $# -ne 0 ]]; then
  fail unexpected_argument
fi

[[ -f "$MAIN" ]] || fail main_missing
[[ -f "$FRONTEND" ]] || fail frontend_missing
[[ -x "$LAUNCHER" ]] || fail launcher_missing
[[ -f "$FIXTURE" ]] || fail fixture_missing
[[ -f "$NO_IMPORT_FIXTURE" ]] || fail no_import_fixture_missing

python3 - "$MAIN" "$FRONTEND" "$LAUNCHER" <<'PY' || exit 1
import re
import sys
from pathlib import Path

main = Path(sys.argv[1]).read_text(encoding="utf-8")
frontend = Path(sys.argv[2]).read_text(encoding="utf-8")
launcher = Path(sys.argv[3]).read_text(encoding="utf-8")

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
    raise AssertionError(f"unterminated_body_{name}")

try:
    native = function_body(main, "run_native_v2_compile_mode")
    assert native.count("compiler_collect_visibility_closure_into(") == 1, "canonical_collection_count"
    assert native.count("module_frontend_compile_collected_to_file(") == 1, "canonical_compile_call_count"
    assert "compiler_visibility_preflight(" not in native, "legacy_preflight_on_canonical_path"
    assert "module_frontend_compile_imported_to_file(" not in native, "legacy_adapter_on_canonical_path"
    assert native.index("compiler_collect_visibility_closure_into(") < native.index("module_frontend_compile_collected_to_file("), "collection_order"
    assert "&!compile_closure" in native and "&!compile_programs" in native, "caller_owned_refs_missing"

    canonical = function_body(frontend, "module_frontend_compile_collected_to_file")
    assert "module_frontend_collect_ast_closure_programs_into(" not in canonical, "canonical_recollects"
    assert "closure: &!ModuleClosure" in frontend, "closure_not_by_ref"
    assert "programs: &! [Program; 256]" in frontend, "programs_not_by_ref"
    assert canonical.index("module_frontend_collected_snapshot_status(") < canonical.index("visibility_begin"), "snapshot_after_visibility"
    assert canonical.index("visibility_begin") < canonical.index("lower_begin"), "lower_before_visibility"
    assert "module_frontend_lower_programs_array_direct_box(programs, &(*closure)" in canonical, "lowering_not_on_collected_inputs"

    legacy = function_body(frontend, "module_frontend_compile_imported_to_file")
    assert legacy.count("module_frontend_collect_ast_closure_programs_into(") == 1, "legacy_collection_count"
    assert legacy.count("module_frontend_compile_collected_to_file(") == 1, "legacy_delegate_count"

    snapshot = function_body(frontend, "module_frontend_closure_snapshot_self_test")
    assert snapshot.count("module_frontend_collect_ast_closure_programs_into(") == 2, "self_test_collection_count"
    assert "closure.collection_id = first_id" in snapshot, "controlled_stale_snapshot_missing"
    assert "stale_status != 1" in snapshot and "current_status != 0" in snapshot, "drift_refusal_assertion_missing"

    snapshot_status = function_body(frontend, "module_frontend_collected_snapshot_status")
    assert "use_path_to_string((*programs)[node_id].module_path)" in snapshot_status, "snapshot_logical_path_not_read_by_value"
    assert "extract_imports_from_ast((*programs)[node_id])" in snapshot_status, "snapshot_imports_not_read_by_value"
    assert "let snapshot_program = (*programs)[node_id]" not in snapshot_status, "snapshot_program_retained_by_value"
    assert "&(*programs)[node_id]" not in snapshot_status, "seed_unsafe_indexed_program_borrow"

    assert re.search(
        r'exec\s+timeout\s+300\s+"\$RAW_MADAROS"\s+build\s+"\$src"\s+-o\s+"\$out"',
        launcher,
    ), "public_launcher_drops_build_verb"
    assert not re.search(
        r'exec\s+timeout\s+300\s+"\$RAW_MADAROS"\s+"\$src"\s+-o\s+"\$out"',
        launcher,
    ), "legacy_native_positional_dispatch_present"
except AssertionError as exc:
    print(f"MADAROS_SINGLE_CLOSURE_COMPILE_FAIL reason=source_contract_{exc}", file=sys.stderr)
    raise SystemExit(1)
PY

if [[ "$SOURCE_ONLY" -eq 1 ]]; then
  printf '%s\n' 'MADAROS_SINGLE_CLOSURE_COMPILE_RECEIPT status=pass evidence=source_contract runtime=not_run entrypoint=bin/madaros native_dispatch=raw_build canonical_collections=1 legacy_adapter=kept snapshot_drift=instrumented snapshot_program_access=by_value'
  exit 0
fi

[[ -n "$RAW_MADAROS" ]] || fail raw_binary_not_set
[[ -n "$EXPECTED_RAW_SHA256" ]] || fail expected_raw_sha256_not_set
[[ -x "$RAW_MADAROS" ]] || fail raw_binary_not_executable
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
RAW_SHA256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
LAUNCHER_SHA256="$(sha256sum "$LAUNCHER" | awk '{print $1}')"
if [[ "$RAW_SHA256" != "$EXPECTED_RAW_SHA256" ]]; then
  fail raw_sha256_mismatch
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-single-closure-compile.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
SNAPSHOT_LOG="$WORK/snapshot.log"
NO_IMPORT_SNAPSHOT_LOG="$WORK/no-import-snapshot.log"
BUILD_LOG="$WORK/build.log"
LAUNCHER_INFO="$WORK/launcher.info"
ELF="$WORK/witness.elf"
STDOUT="$WORK/stdout"

MADAROS_RAW_BIN="$RAW_MADAROS" "$LAUNCHER" info >"$LAUNCHER_INFO"
grep -Fxq "raw_elf:      $RAW_MADAROS" "$LAUNCHER_INFO" || fail launcher_raw_identity_mismatch

run_snapshot_self_test() {
  local label="$1"
  local fixture="$2"
  local log="$3"
  local rc=0

  set +e
  SOUNIO_MODULE_CLOSURE_TRACE=1 timeout 180 "$RAW_MADAROS" \
    --module-closure-snapshot-self-test "$fixture" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 0 ]] || fail "${label}_snapshot_self_test_rc"
  [[ "$(grep -c '^module_closure_collect: phase=begin ' "$log" || true)" -eq 2 ]] || fail "${label}_snapshot_collection_count"
  grep -Eq '^MODULE_CLOSURE_SNAPSHOT_SELF_TEST status=pass first_id=[1-9][0-9]* second_id=[1-9][0-9]* stale_state=superseded current_state=current collections=2$' "$log" || fail "${label}_snapshot_receipt_missing"
  grep -Fxq 'module-closure-snapshot-self-test: OK' "$log" || fail "${label}_snapshot_main_receipt_missing"
}

run_snapshot_self_test no_import "$NO_IMPORT_FIXTURE" "$NO_IMPORT_SNAPSHOT_LOG"
run_snapshot_self_test imported "$FIXTURE" "$SNAPSHOT_LOG"

rm -f "$ELF"
set +e
SOUNIO_MODULE_CLOSURE_TRACE=1 MADAROS_RAW_BIN="$RAW_MADAROS" timeout 360 "$LAUNCHER" \
  --science-boundary off build "$FIXTURE" -o "$ELF" >"$BUILD_LOG" 2>&1
BUILD_RC=$?
set -e
[[ "$BUILD_RC" -eq 0 ]] || fail build_rc
[[ -s "$ELF" ]] || fail runtime_elf_missing
[[ "$(grep -c '^module_closure_collect: phase=begin ' "$BUILD_LOG" || true)" -eq 1 ]] || fail canonical_collection_count
grep -Fq 'imported_compile: collected_begin collection_id=1' "$BUILD_LOG" || fail collected_receipt_missing
grep -Fq 'imported_compile: visibility_begin' "$BUILD_LOG" || fail visibility_not_reached
grep -Fq 'imported_compile: lower_begin' "$BUILD_LOG" || fail lowering_not_reached
! grep -Fq 'imported_compile: legacy_adapter_' "$BUILD_LOG" || fail legacy_adapter_used
! grep -Fq 'imported_compile: snapshot_invalid' "$BUILD_LOG" || fail snapshot_invalid

set +e
timeout 30 "$ELF" >"$STDOUT" 2>&1
ELF_RC=$?
set -e
[[ "$ELF_RC" -eq 0 ]] || fail elf_rc
printf '42\n' | cmp -s - "$STDOUT" || fail elf_stdout

printf 'MADAROS_SINGLE_CLOSURE_COMPILE_RECEIPT status=pass evidence=source_contract_and_pinned_runtime entrypoint=bin/madaros launcher_sha256=%s raw_sha256=%s launcher_raw_identity=pinned native_dispatch=raw_build canonical_collections=1 snapshot_shapes=no_import+imported snapshot_collections=4 snapshot_drift=refused snapshot_program_access=by_value visibility=same_snapshot lowering=same_snapshot runtime_stdout=42_LF legacy_adapter=kept recollection_fallback=none\n' \
  "$LAUNCHER_SHA256" "$RAW_SHA256"
