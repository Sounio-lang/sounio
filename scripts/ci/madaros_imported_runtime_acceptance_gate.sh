#!/usr/bin/env bash
# #901: imported nominal layout identity must survive lowering and fail closed.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
EXPECTED_SHA="${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_EXPECTED_SHA256:-}"
KEEP_WORK="${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_KEEP:-0}"
WORK="${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_DIR:-}"
PASS_SOURCE="$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_nested_field_chain_main.sio"
MISS_SOURCE="$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_known_layout_miss_main.sio"
METHOD_MISS_SOURCE="$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_method_result_known_layout_miss_main.sio"
INDEX_MISS_SOURCE="$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_indexed_element_known_layout_miss_main.sio"

fail() { echo "[madaros-imported-runtime-acceptance] FAIL: $*" >&2; exit 1; }
sha256() { sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'; }
elf_magic() { od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' \n'; }
elf_u16() { od -An -tu2 -j "$2" -N2 "$1" 2>/dev/null | tr -d ' \n'; }

assert_elf() {
  local path="$1" label="$2"
  [[ -s "$path" ]] || fail "$label is missing or empty: $path"
  [[ "$(elf_magic "$path")" == 7f454c46 ]] || fail "$label is not ELF"
  [[ "$(elf_u16 "$path" 16)" == 2 ]] || fail "$label is not ET_EXEC"
  [[ "$(elf_u16 "$path" 18)" == 62 ]] || fail "$label is not x86-64"
}

assert_no_fallback() {
  local log="$1"
  if grep -Eiq 'native_prebundle:|falling back to full IR path|specialized lower failed|multi-mod fallback|compact modular IR table path|legacy compact IR differential enabled' "$log"; then
    cat "$log" >&2
    fail "fallback marker observed in $log"
  fi
}

[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit Madaros ELF'
assert_elf "$RAW_MADAROS" 'Madaros input'
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
RAW_SHA="$(sha256 "$RAW_MADAROS")"
[[ -z "$EXPECTED_SHA" || "$RAW_SHA" == "$EXPECTED_SHA" ]] || fail "compiler SHA mismatch expected=$EXPECTED_SHA actual=$RAW_SHA"
for source in "$PASS_SOURCE" "$MISS_SOURCE" "$METHOD_MISS_SOURCE" "$INDEX_MISS_SOURCE"; do
  [[ -f "$source" ]] || fail "acceptance witness missing: $source"
done

if [[ -n "$WORK" ]]; then
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-imported-runtime-acceptance.XXXXXX)"
fi
mkdir -p "$WORK/pass"
[[ "$KEEP_WORK" == 1 ]] || trap 'rm -rf "$WORK"' EXIT

run_check() {
  local source="$1" log="$2"
  set +e
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$RAW_MADAROS" --check "$source" >"$log" 2>&1
  CHECK_RC=$?
  set -e
}

run_compile() {
  local source="$1" output="$2" log="$3"
  rm -f "$output"
  set +e
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$RAW_MADAROS" --native-v2-compile "$source" -o "$output" >"$log" 2>&1
  COMPILE_RC=$?
  set -e
}

PASS_ELF="$WORK/pass/witness.elf"
run_check "$PASS_SOURCE" "$WORK/pass/check.log"
[[ "$CHECK_RC" -eq 0 ]] || { cat "$WORK/pass/check.log" >&2; fail "positive checker failed rc=$CHECK_RC"; }
run_compile "$PASS_SOURCE" "$PASS_ELF" "$WORK/pass/compile.log"
[[ "$COMPILE_RC" -eq 0 ]] || { cat "$WORK/pass/compile.log" >&2; fail "positive compile failed rc=$COMPILE_RC"; }
assert_no_fallback "$WORK/pass/compile.log"
assert_elf "$PASS_ELF" 'positive witness'
chmod +x "$PASS_ELF"
set +e
"$PASS_ELF" >"$WORK/pass/runtime.log" 2>&1
RUNTIME_RC=$?
set -e
[[ "$RUNTIME_RC" -eq 0 ]] || { cat "$WORK/pass/runtime.log" >&2; fail "positive ELF rc=$RUNTIME_RC"; }
[[ "$(grep -Fxc 520 "$WORK/pass/runtime.log")" -eq 2 ]] || { cat "$WORK/pass/runtime.log" >&2; fail 'direct and materialized values were not both 520'; }
grep -Fxq 'ISSUE_901_NESTED_FIELD_CHAIN_OK' "$WORK/pass/runtime.log" || fail 'positive marker absent'
grep -Fxq 'ISSUE_901_NOMINAL_PROJECTION_CONTROLS_OK' "$WORK/pass/runtime.log" || fail 'projection control marker absent'

# This is intentionally a lowering-boundary probe: the invalid program must
# pass today's checker and demonstrate that lowering itself refuses the miss.
# A future checker-owned rejection should replace this witness contract rather
# than being silently counted as the same proof.
run_known_layout_miss() {
  local label="$1" source="$2" type_name="$3" case_dir="$WORK/miss-$1" elf="$WORK/miss-$1/witness.elf"
  mkdir -p "$case_dir"
  run_check "$source" "$case_dir/check.log"
  [[ "$CHECK_RC" -eq 0 ]] || { cat "$case_dir/check.log" >&2; fail "$label negative did not reach lowering: checker rc=$CHECK_RC"; }
  run_compile "$source" "$elf" "$case_dir/compile.log"
  [[ "$COMPILE_RC" -ne 0 ]] || { cat "$case_dir/compile.log" >&2; fail "$label known-layout miss unexpectedly compiled"; }
  [[ ! -e "$elf" ]] || fail "$label known-layout miss emitted an ELF"
  assert_no_fallback "$case_dir/compile.log"
  grep -Fxq "LOWER_NOMINAL_LAYOUT_MISS type=$type_name field=family_id" "$case_dir/compile.log" \
    || { cat "$case_dir/compile.log" >&2; fail "$label exact nominal miss marker absent"; }
  for marker in \
    'imported_compile: typecheck ok' \
    'imported_compile: lower_begin' \
    'imported_compile: lower_done' \
    'IR lowering failed during merge:'; do
    grep -Fq "$marker" "$case_dir/compile.log" || { cat "$case_dir/compile.log" >&2; fail "$label negative causality marker absent: $marker"; }
  done
}

run_known_layout_miss materialized "$MISS_SOURCE" InnerState
run_known_layout_miss method-result "$METHOD_MISS_SOURCE" MethodInnerState
run_known_layout_miss indexed-element "$INDEX_MISS_SOURCE" IndexedInnerState

echo "[madaros-imported-runtime-acceptance] PASS raw_sha256=$RAW_SHA direct=520 materialized=520 method_field=6102 indexed_field=6102 known_layout_miss=materialized,method-result,indexed-element_refused fallback=0"
