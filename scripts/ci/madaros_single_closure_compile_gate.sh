#!/usr/bin/env bash
# Prove that the canonical native-v2 source path compiles one collected closure.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="$ROOT_DIR/self-hosted/compiler/main.sio"
FRONTEND="$ROOT_DIR/self-hosted/compiler/module_frontend.sio"
FIXTURE="$ROOT_DIR/tests/compiler/module_closure_single_compile/main.sio"
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
[[ -f "$FIXTURE" ]] || fail fixture_missing

python3 - "$MAIN" "$FRONTEND" <<'PY' || exit 1
import re
import sys
from pathlib import Path

main = Path(sys.argv[1]).read_text(encoding="utf-8")
frontend = Path(sys.argv[2]).read_text(encoding="utf-8")

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
except AssertionError as exc:
    print(f"MADAROS_SINGLE_CLOSURE_COMPILE_FAIL reason=source_contract_{exc}", file=sys.stderr)
    raise SystemExit(1)
PY

if [[ "$SOURCE_ONLY" -eq 1 ]]; then
  printf '%s\n' 'MADAROS_SINGLE_CLOSURE_COMPILE_RECEIPT status=pass evidence=source_contract runtime=not_run canonical_collections=1 legacy_adapter=kept snapshot_drift=instrumented'
  exit 0
fi

[[ -n "$RAW_MADAROS" ]] || fail raw_binary_not_set
[[ -x "$RAW_MADAROS" ]] || fail raw_binary_not_executable
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
RAW_SHA256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
if [[ -n "$EXPECTED_RAW_SHA256" && "$RAW_SHA256" != "$EXPECTED_RAW_SHA256" ]]; then
  fail raw_sha256_mismatch
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-single-closure-compile.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
SNAPSHOT_LOG="$WORK/snapshot.log"
BUILD_LOG="$WORK/build.log"
ELF="$WORK/witness.elf"
STDOUT="$WORK/stdout"

set +e
SOUNIO_MODULE_CLOSURE_TRACE=1 timeout 180 "$RAW_MADAROS" \
  --module-closure-snapshot-self-test "$FIXTURE" >"$SNAPSHOT_LOG" 2>&1
SNAPSHOT_RC=$?
set -e
[[ "$SNAPSHOT_RC" -eq 0 ]] || fail snapshot_self_test_rc
[[ "$(grep -c '^module_closure_collect: phase=begin ' "$SNAPSHOT_LOG" || true)" -eq 2 ]] || fail snapshot_collection_count
grep -Eq '^MODULE_CLOSURE_SNAPSHOT_SELF_TEST status=pass first_id=[1-9][0-9]* second_id=[1-9][0-9]* stale_state=superseded current_state=current collections=2$' "$SNAPSHOT_LOG" || fail snapshot_receipt_missing
grep -Fxq 'module-closure-snapshot-self-test: OK' "$SNAPSHOT_LOG" || fail snapshot_main_receipt_missing

rm -f "$ELF"
set +e
SOUNIO_MODULE_CLOSURE_TRACE=1 timeout 360 "$RAW_MADAROS" \
  build "$FIXTURE" -o "$ELF" >"$BUILD_LOG" 2>&1
BUILD_RC=$?
set -e
[[ "$BUILD_RC" -eq 0 ]] || fail build_rc
[[ -s "$ELF" ]] || fail fresh_elf_missing
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

printf 'MADAROS_SINGLE_CLOSURE_COMPILE_RECEIPT status=pass evidence=source_and_runtime raw_sha256=%s canonical_collections=1 snapshot_collections=2 snapshot_drift=refused visibility=same_snapshot lowering=same_snapshot runtime_stdout=42_LF legacy_adapter=kept recollection_fallback=none\n' \
  "$RAW_SHA256"
