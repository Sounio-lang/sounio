#!/usr/bin/env bash
# Source-to-IR acceptance for typed ordered-path and grouping provenance.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPLICIT_SOUC="${SOUNIO_ORDERED_PATH_COMPILER_BIN:-}"
SOUC="${EXPLICIT_SOUC:-$ROOT_DIR/bin/souc}"
EXPECTED_COMPILER_SHA256="${SOUNIO_ORDERED_PATH_EXPECTED_COMPILER_SHA256:-}"
EXPECTED_SOURCE_SHA="${SOUNIO_ORDERED_PATH_EXPECTED_SOURCE_SHA:-}"
REQUIRE_COMPILER="${SOUNIO_ORDERED_PATH_REQUIRE_COMPILER:-0}"
KEEP_WORK="${SOUNIO_ORDERED_PATH_KEEP:-0}"
IR_MODEL="$ROOT_DIR/self-hosted/ir/ir.sio"
FRONTEND="$ROOT_DIR/self-hosted/compiler/module_frontend.sio"
CHECKER_BRIDGE="$ROOT_DIR/self-hosted/check/mod.sio"
CONCEPT="$ROOT_DIR/docs/internal/concepts/ordered-path-provenance.md"
REGISTRY="$ROOT_DIR/docs/internal/concepts/registry.tsv"
BINDINGS="$ROOT_DIR/docs/internal/concepts/bindings.tsv"
POSITIVE="$ROOT_DIR/tests/compiler/ordered_path_provenance_source.sio"
IMPORTED_MAIN="$ROOT_DIR/tests/compiler/ordered_path_provenance_imported_main.sio"
IMPORTED_LEAF="$ROOT_DIR/tests/compiler/ordered_path_provenance_imported_leaf.sio"
AB_REJECT="$ROOT_DIR/tests/compile-fail/ordered_path_ab_cannot_replace_ba.sio"
GROUPING_REJECT="$ROOT_DIR/tests/compile-fail/ordered_path_left_cannot_replace_right.sio"
CATEGORY_REJECT="$ROOT_DIR/tests/compile-fail/ordered_path_order_witness_cannot_replace_nonassociativity.sio"
OBSERVATION_REJECT="$ROOT_DIR/tests/compile-fail/ordered_path_occupancy_cannot_replace_state.sio"
EXPECTED_RUNTIME='ORDERED_PATH_OK occupancy=800 input=333 immediate=877 later_ab=874 later_ba=879 later_left=5 later_right=866'

fail() {
  echo "[ordered-path-source-ir] FAIL: $*" >&2
  exit 1
}

require_text() {
  local pattern="$1"
  local file="$2"
  rg -q "$pattern" "$file" || fail "missing required anchor '$pattern' in $file"
}

reject_text() {
  local pattern="$1"
  local file="$2"
  if rg -q "$pattern" "$file"; then
    fail "forbidden anchor '$pattern' found in $file"
  fi
}

case "$REQUIRE_COMPILER" in
  0|1) ;;
  *) fail "SOUNIO_ORDERED_PATH_REQUIRE_COMPILER must be 0 or 1" ;;
esac

for path in \
  "$IR_MODEL" "$FRONTEND" "$CHECKER_BRIDGE" "$CONCEPT" "$REGISTRY" "$BINDINGS" "$POSITIVE" \
  "$IMPORTED_MAIN" "$IMPORTED_LEAF" "$AB_REJECT" "$GROUPING_REJECT" \
  "$CATEGORY_REJECT" "$OBSERVATION_REJECT"; do
  [[ -f "$path" ]] || fail "required source is missing: $path"
done
[[ -x "$SOUC" ]] || fail "compiler is missing or not executable: $SOUC"

require_text 'SOUNIO_ORDERED_PATH_TRACE' "$FRONTEND"
require_text '^fn module_frontend_print_ordered_path_receipt' "$FRONTEND"
require_text 'print\(" authority=0\\n"\)' "$FRONTEND"
require_text 'module_frontend_print_ordered_path_receipt\("single-final", optimize' "$FRONTEND"
require_text 'module_frontend_print_ordered_path_receipt\("merged-final", optimize' "$FRONTEND"
require_text '^pub fn ir_ontology_parameter_link_total' "$IR_MODEL"
require_text '^pub fn ir_ontology_parameter_link_count_is_bounded' "$IR_MODEL"
require_text '^pub fn ir_ontology_parameter_link_function_name_at' "$IR_MODEL"
require_text '^pub fn ir_ontology_parameter_link_parameter_index_at' "$IR_MODEL"
require_text '^pub fn ir_ontology_parameter_link_class_name_at' "$IR_MODEL"
require_text 'audit-only accessors' "$IR_MODEL"
require_text 'checker_apply_ir_ontology_parameter_links_from_items' "$CHECKER_BRIDGE"
require_text 'Concept-ID: `SOUNIO-ORDERED-PATH-PROVENANCE`' "$CONCEPT"
require_text '^SOUNIO-ORDERED-PATH-PROVENANCE[[:space:]]+executable[[:space:]]' "$REGISTRY"
require_text '^SOUNIO-ORDERED-PATH-PROVENANCE[[:space:]]+self-hosted/ir/ir.sio[[:space:]]+ordered-signature-identity$' "$BINDINGS"

single_cleanup_line="$(rg -n 'let cleanup_receipt = opt_cleanup_module_inplace\(&! \(\*module_box\)\)' "$FRONTEND" | cut -d: -f1)"
single_trace_line="$(rg -n 'module_frontend_print_ordered_path_receipt\("single-final"' "$FRONTEND" | cut -d: -f1)"
merged_cleanup_line="$(rg -n 'let cleanup_receipt = opt_cleanup_module_inplace\(&! merged_module\)' "$FRONTEND" | cut -d: -f1)"
merged_trace_line="$(rg -n 'module_frontend_print_ordered_path_receipt\("merged-final"' "$FRONTEND" | cut -d: -f1)"
[[ -n "$single_cleanup_line" && -n "$single_trace_line" && "$single_trace_line" -gt "$single_cleanup_line" ]] ||
  fail "single-module ordered-path trace must run after cleanup"
[[ -n "$merged_cleanup_line" && -n "$merged_trace_line" && "$merged_trace_line" -gt "$merged_cleanup_line" ]] ||
  fail "merged ordered-path trace must run after final cleanup"

require_text 'fn state_after_ab' "$POSITIVE"
require_text 'occupancy: ReceptorOccupancyObservation' "$POSITIVE"
require_text 'a: StepA' "$POSITIVE"
require_text 'b: StepB' "$POSITIVE"
require_text 'order: OrderABReceipt' "$POSITIVE"
require_text 'fn state_after_ba' "$POSITIVE"
require_text 'order: OrderBAReceipt' "$POSITIVE"
require_text 'fn left_grouped_abc' "$POSITIVE"
require_text 'grouping: LeftGrouping' "$POSITIVE"
require_text 'fn right_grouped_abc' "$POSITIVE"
require_text 'grouping: RightGrouping' "$POSITIVE"
require_text 'let mismatch =' "$POSITIVE"
require_text '\(immediate_ab \^ 877\)' "$POSITIVE"
require_text '\(later_left \^ 5\)' "$POSITIVE"
require_text '^use ordered_path_provenance_imported_leaf::' "$IMPORTED_MAIN"
require_text 'pub fn imported_ordered_path_run' "$IMPORTED_LEAF"

if [[ -n "${SOUNIO_ORDERED_PATH_WORK_DIR:-}" ]]; then
  WORK="$SOUNIO_ORDERED_PATH_WORK_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing work directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-ordered-path-source-ir.XXXXXX)"
fi
if [[ "$KEEP_WORK" != 1 ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

for src in "$POSITIVE" "$IMPORTED_MAIN"; do
  label="$(basename "$src" .sio)"
  if ! "$SOUC" check "$src" >"$WORK/$label.check.log" 2>&1; then
    cat "$WORK/$label.check.log" >&2
    fail "source preflight failed: $src"
  fi
done

check_rejection() {
  local label="$1"
  local src="$2"
  local expected="$3"
  local found="$4"
  local log="$WORK/$label.check.log"
  if "$SOUC" check "$src" >"$log" 2>&1; then
    fail "$label unexpectedly type checked"
  fi
  require_text 'error\[E009' "$log"
  require_text "$expected" "$log"
  require_text "$found" "$log"
}

check_rejection ab-state "$AB_REJECT" 'expected StateAfterBA' 'found StateAfterAB'
check_rejection grouping-state "$GROUPING_REJECT" 'expected RightGroupedABCState' 'found LeftGroupedABCState'
check_rejection witness-category "$CATEGORY_REJECT" 'expected NonAssociativityWitness' 'found OrderSensitivityWitness'
check_rejection observation-state "$OBSERVATION_REJECT" 'expected FunctionalState' 'found ReceptorOccupancyObservation'

compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
source_sha="$(git -C "$ROOT_DIR" rev-parse HEAD)"

if [[ "$REQUIRE_COMPILER" != 1 ]]; then
  echo "[ordered-path-source-ir] LOCAL_PREFLIGHT_ONLY source_preflight=2 category_rejections=4 compiler_state=source-check-only compiler_sha256=$compiler_sha256 source_sha=$source_sha merge_ready=0"
  exit 0
fi

[[ -n "$EXPLICIT_SOUC" ]] || fail "strict mode requires SOUNIO_ORDERED_PATH_COMPILER_BIN"
[[ "$EXPECTED_COMPILER_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "strict mode requires a lowercase 64-hex SOUNIO_ORDERED_PATH_EXPECTED_COMPILER_SHA256"
[[ "$compiler_sha256" == "$EXPECTED_COMPILER_SHA256" ]] ||
  fail "compiler SHA-256 mismatch: expected=$EXPECTED_COMPILER_SHA256 actual=$compiler_sha256"
[[ "$EXPECTED_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] ||
  fail "strict mode requires a lowercase 40-hex SOUNIO_ORDERED_PATH_EXPECTED_SOURCE_SHA"
[[ "$source_sha" == "$EXPECTED_SOURCE_SHA" ]] ||
  fail "source Git SHA mismatch: expected=$EXPECTED_SOURCE_SHA actual=$source_sha"
[[ -z "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=no)" ]] ||
  fail "strict mode requires a clean tracked source worktree"

compile_source() {
  local label="$1"
  local src="$2"
  local elf="$WORK/$label.elf"
  local log="$WORK/$label.compile.log"
  SOUNIO_ORDERED_PATH_TRACE=1 MADAROS_RAW_BIN="$SOUC" SOUNIO_SOUC_ENGINE=madaros \
    "$ROOT_DIR/bin/souc" -O "$src" -o "$elf" >"$log" 2>&1 || {
      cat "$log" >&2
      fail "$label failed to compile"
    }
  reject_text 'falling back|lean_single' "$log"
}

require_link() {
  local label="$1"
  local phase="$2"
  local function_name="$3"
  local parameter_index="$4"
  local class_name="$5"
  local log="$WORK/$label.compile.log"
  local pattern="^\\[ordered-path-link\\] phase=$phase schema=1 function=$function_name parameter=$parameter_index class=$class_name authority=0$"
  [[ "$(rg -c "$pattern" "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label did not preserve exactly one $function_name/$parameter_index/$class_name link"
  }
}

require_function_link_count() {
  local label="$1"
  local phase="$2"
  local function_name="$3"
  local expected="$4"
  local log="$WORK/$label.compile.log"
  local pattern="^\\[ordered-path-link\\] phase=$phase schema=1 function=$function_name "
  [[ "$(rg -c "$pattern" "$log")" -eq "$expected" ]] || {
    cat "$log" >&2
    fail "$label emitted the wrong link count for $function_name"
  }
}

require_trace_matrix() {
  local label="$1"
  local phase="$2"
  local log="$WORK/$label.compile.log"
  require_text "^\\[ordered-path-native-v2\\] phase=$phase schema=1 optimize=1 functions=[0-9]+ links=26 bounded=1 authority=0$" "$log"
  [[ "$(rg -c "^\\[ordered-path-link\\] phase=$phase " "$log")" -eq 26 ]] ||
    fail "$label did not emit exactly 26 ordered signature links"

  require_function_link_count "$label" "$phase" state_after_ab 4
  require_link "$label" "$phase" state_after_ab 1 ReceptorOccupancyObservation
  require_link "$label" "$phase" state_after_ab 2 StepA
  require_link "$label" "$phase" state_after_ab 3 StepB
  require_link "$label" "$phase" state_after_ab 4 OrderABReceipt

  require_function_link_count "$label" "$phase" state_after_ba 4
  require_link "$label" "$phase" state_after_ba 1 ReceptorOccupancyObservation
  require_link "$label" "$phase" state_after_ba 2 StepB
  require_link "$label" "$phase" state_after_ba 3 StepA
  require_link "$label" "$phase" state_after_ba 4 OrderBAReceipt

  require_function_link_count "$label" "$phase" left_grouped_abc 5
  require_link "$label" "$phase" left_grouped_abc 1 ReceptorOccupancyObservation
  require_link "$label" "$phase" left_grouped_abc 2 StepA
  require_link "$label" "$phase" left_grouped_abc 3 StepB
  require_link "$label" "$phase" left_grouped_abc 4 StepC
  require_link "$label" "$phase" left_grouped_abc 5 LeftGrouping

  require_function_link_count "$label" "$phase" right_grouped_abc 5
  require_link "$label" "$phase" right_grouped_abc 1 ReceptorOccupancyObservation
  require_link "$label" "$phase" right_grouped_abc 2 StepA
  require_link "$label" "$phase" right_grouped_abc 3 StepB
  require_link "$label" "$phase" right_grouped_abc 4 StepC
  require_link "$label" "$phase" right_grouped_abc 5 RightGrouping

  require_link "$label" "$phase" project_ab 0 StateAfterAB
  require_link "$label" "$phase" project_ba 0 StateAfterBA
  require_link "$label" "$phase" continue_ab 0 StateAfterAB
  require_link "$label" "$phase" continue_ba 0 StateAfterBA
  require_link "$label" "$phase" project_left 0 LeftGroupedABCState
  require_link "$label" "$phase" project_right 0 RightGroupedABCState
  require_link "$label" "$phase" continue_left 0 LeftGroupedABCState
  require_link "$label" "$phase" continue_right 0 RightGroupedABCState
}

run_elf() {
  local label="$1"
  local elf="$WORK/$label.elf"
  local log="$WORK/$label.runtime.log"
  local magic
  local rc=0
  [[ -s "$elf" ]] || fail "$label did not produce a non-empty ELF"
  magic="$(od -An -tx1 -N4 "$elf" | tr -d ' \n')"
  [[ "$magic" == 7f454c46 ]] || fail "$label produced a non-ELF artifact: magic=$magic"
  chmod +x "$elf"
  set +e
  timeout 10 "$elf" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 0 ]] || {
    cat "$log" >&2
    fail "$label runtime witness returned rc=$rc"
  }
  [[ "$(cat "$log")" == "$EXPECTED_RUNTIME" ]] || {
    cat "$log" >&2
    fail "$label runtime output did not match the exact scalar-collision witness"
  }
}

compile_source single "$POSITIVE"
require_trace_matrix single single-final
run_elf single

compile_source imported "$IMPORTED_MAIN"
require_text 'module_native_driver: -O selects finalized full IR cleanup path' "$WORK/imported.compile.log"
require_trace_matrix imported merged-final
run_elf imported

echo "[ordered-path-source-ir] PASS: source_preflight=2 category_rejections=4 exact_immediate_collision=877 order_links=AB,BA grouping_links=left,right state_links=8 single_runtime=1 imported_runtime=1 fallback=0 compiler_sha256=$compiler_sha256 source_sha=$source_sha merge_ready=1"
