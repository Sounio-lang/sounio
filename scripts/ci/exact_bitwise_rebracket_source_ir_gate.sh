#!/usr/bin/env bash
# Source-to-IR acceptance for dominated exact bitwise rebracketing.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPLICIT_SOUC="${SOUNIO_REBRACKET_COMPILER_BIN:-}"
SOUC="${EXPLICIT_SOUC:-$ROOT_DIR/bin/souc}"
EXPECTED_COMPILER_SHA256="${SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256:-}"
REQUIRE_COMPILER="${SOUNIO_REBRACKET_REQUIRE_COMPILER:-0}"
KEEP_WORK="${SOUNIO_REBRACKET_KEEP:-0}"
AUTHORITY_GATE="$ROOT_DIR/scripts/ci/exact_bitwise_rebracket_authority_gate.sh"
OPT_CLEANUP="$ROOT_DIR/self-hosted/ir/opt_cleanup.sio"
IR_MODEL="$ROOT_DIR/self-hosted/ir/ir.sio"
CHECKER_BRIDGE="$ROOT_DIR/self-hosted/check/mod.sio"
POSITIVE="$ROOT_DIR/tests/compiler/rebracket_authority_cross_block_source.sio"
IMPORTED_POSITIVE="$ROOT_DIR/tests/compiler/rebracket_authority_cross_block_imported_main.sio"
IMPORTED_LEAF="$ROOT_DIR/tests/compiler/rebracket_authority_cross_block_imported_leaf.sio"
NONDOMINATING="$ROOT_DIR/tests/compiler/rebracket_authority_nondominating_source.sio"
LOOP_REFUSAL="$ROOT_DIR/tests/compiler/rebracket_authority_loop_refusal_source.sio"
ONTOLOGY_REJECT="$ROOT_DIR/tests/compile-fail/rebracket_authority_unrelated_ontology_obligation.sio"
ONTOLOGY_CLASS_HASH="7199034902620903764"

fail() {
  echo "[rebracket-source-ir] FAIL: $*" >&2
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
  *) fail "SOUNIO_REBRACKET_REQUIRE_COMPILER must be 0 or 1" ;;
esac

for path in "$AUTHORITY_GATE" "$OPT_CLEANUP" "$IR_MODEL" "$CHECKER_BRIDGE" "$POSITIVE" "$IMPORTED_POSITIVE" "$IMPORTED_LEAF" "$NONDOMINATING" "$LOOP_REFUSAL" "$ONTOLOGY_REJECT"; do
  [[ -f "$path" ]] || fail "required source is missing: $path"
done
[[ -x "$SOUC" ]] || fail "compiler is missing or not executable: $SOUC"

require_text 'let inner = x & 255' "$POSITIVE"
require_text 'class ExactBitwiseRebracketObligation subclass_of CompilerSemanticObligation' "$POSITIVE"
require_text 'obligation: ExactBitwiseRebracketObligation' "$POSITIVE"
require_text 'let branch_value = if choose_left' "$POSITIVE"
require_text '\(inner & 15\) \| branch_zero' "$POSITIVE"
require_text '^use rebracket_authority_cross_block_imported_leaf::' "$IMPORTED_POSITIVE"
require_text 'imported_rebracket_run\(\)' "$IMPORTED_POSITIVE"
require_text 'class ExactBitwiseRebracketObligation subclass_of CompilerSemanticObligation' "$IMPORTED_LEAF"
require_text 'obligation: ExactBitwiseRebracketObligation' "$IMPORTED_LEAF"
require_text 'let inner = x & 255' "$IMPORTED_LEAF"
require_text '\(inner & 15\) \| branch_zero' "$IMPORTED_LEAF"
require_text 'let branch_value = if choose_left \{ x & 255 \} else \{ x \}' "$NONDOMINATING"
require_text 'while i < rounds' "$LOOP_REFUSAL"
require_text '^    0$' "$LOOP_REFUSAL"
require_text 'let observation: ReceptorOccupancyObservation' "$ONTOLOGY_REJECT"
require_text 'checker_apply_ir_ontology_parameter_links_from_items' "$CHECKER_BRIDGE"
require_text 'This helper composes only audit links' "$IR_MODEL"
require_text 'current\.parameter_index == candidate\.parameter_index' "$IR_MODEL"
require_text 'ir_name_eq\(current\.function_name, candidate\.function_name\)' "$IR_MODEL"
require_text 'ir_name_eq\(current\.class_name, candidate\.class_name\)' "$IR_MODEL"
require_text 'assert\(ir_name_eq\(function_name, \(\*module\)\.functions\[fi as usize\]\.name\)\)' "$OPT_CLEANUP"

[[ "$(rg -c 'ir_ontology_parameter_link_count_for_function' "$OPT_CLEANUP")" -eq 1 ]] ||
  fail "ontology parameter-link count must have exactly one audit-only optimizer consumer"
[[ "$(rg -c 'ir_ontology_parameter_last_class_hash_for_function' "$OPT_CLEANUP")" -eq 1 ]] ||
  fail "ontology class hash must have exactly one audit-only optimizer consumer"
cleanup_call_line="$(rg -n '\(\*module\)\.functions\[fi as usize\] = opt_cleanup_function_with_algebras_and_audit' "$OPT_CLEANUP" | cut -d: -f1)"
name_stability_line="$(rg -n 'assert\(ir_name_eq\(function_name, \(\*module\)\.functions\[fi as usize\]\.name\)\)' "$OPT_CLEANUP" | cut -d: -f1)"
ontology_query_line="$(rg -n 'let link_count = ir_ontology_parameter_link_count_for_function' "$OPT_CLEANUP" | cut -d: -f1)"
[[ -n "$cleanup_call_line" && -n "$name_stability_line" && -n "$ontology_query_line" &&
   "$name_stability_line" -gt "$cleanup_call_line" && "$ontology_query_line" -gt "$name_stability_line" ]] ||
  fail "ontology identity must be queried only after the authority-owned cleanup decision"

if [[ -n "${SOUNIO_REBRACKET_SOURCE_WORK_DIR:-}" ]]; then
  WORK="$SOUNIO_REBRACKET_SOURCE_WORK_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing work directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-rebracket-source-ir.XXXXXX)"
fi
if [[ "$KEEP_WORK" != 1 ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

for src in "$POSITIVE" "$IMPORTED_POSITIVE" "$NONDOMINATING" "$LOOP_REFUSAL"; do
  label="$(basename "$src" .sio)"
  if ! "$SOUC" check "$src" >"$WORK/$label.check.log" 2>&1; then
    cat "$WORK/$label.check.log" >&2
    fail "source preflight failed: $src"
  fi
done

if "$SOUC" check "$ONTOLOGY_REJECT" >"$WORK/ontology-reject.check.log" 2>&1; then
  fail "unrelated ontology observation discharged a compiler semantic obligation"
fi
require_text 'expected CompilerSemanticObligation' "$WORK/ontology-reject.check.log"
require_text 'found ReceptorOccupancyObservation' "$WORK/ontology-reject.check.log"

compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
source_sha="$(git -C "$ROOT_DIR" rev-parse HEAD)"

if [[ "$REQUIRE_COMPILER" != 1 ]]; then
  echo "[rebracket-source-ir] LOCAL_EVIDENCE_PASS source_preflight=4 ontology_rejection=1 compiler_state=source-check-only compiler_sha256=$compiler_sha256 source_sha=$source_sha merge_ready=0"
  exit 0
fi

[[ -n "$EXPLICIT_SOUC" ]] || fail "strict mode requires SOUNIO_REBRACKET_COMPILER_BIN"
[[ "$EXPECTED_COMPILER_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "strict mode requires a lowercase 64-hex SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256"
[[ "$compiler_sha256" == "$EXPECTED_COMPILER_SHA256" ]] ||
  fail "compiler SHA-256 mismatch: expected=$EXPECTED_COMPILER_SHA256 actual=$compiler_sha256"
[[ -z "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=no)" ]] ||
  fail "strict mode requires a clean tracked source worktree"

SOUNIO_REBRACKET_COMPILER_BIN="$SOUC" \
SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256="$EXPECTED_COMPILER_SHA256" \
SOUNIO_REBRACKET_REQUIRE_COMPILER=1 \
  bash "$AUTHORITY_GATE" >"$WORK/authority-gate.log" 2>&1 || {
    cat "$WORK/authority-gate.log" >&2
    fail "inherited authority gate failed"
  }
require_text 'compiler_state=executable' "$WORK/authority-gate.log"
require_text 'merge_ready=1' "$WORK/authority-gate.log"

compile_source() {
  local label="$1"
  local mode="$2"
  local src="$3"
  local elf="$WORK/$label.elf"
  local log="$WORK/$label.compile.log"
  local -a optimize_args=()
  if [[ "$mode" == optimized ]]; then
    optimize_args=(-O)
  else
    optimize_args=(-t native)
  fi
  SOUNIO_REBRACKET_TRACE=1 MADAROS_RAW_BIN="$SOUC" SOUNIO_SOUC_ENGINE=madaros \
    "$ROOT_DIR/bin/souc" "${optimize_args[@]}" "$src" -o "$elf" >"$log" 2>&1 || {
      cat "$log" >&2
      fail "$label failed to compile"
    }
  reject_text 'falling back|lean_single' "$log"
}

require_receipt() {
  local label="$1"
  local pattern="$2"
  local log="$WORK/$label.compile.log"
  rg -q "$pattern" "$log" || {
    cat "$log" >&2
    fail "$label omitted its exact source-to-IR receipt"
  }
  [[ "$(rg -c '^\[rebracket-native-v2\]' "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label emitted an ambiguous number of receipts"
  }
}

run_elf_expect_zero() {
  local label="$1"
  local elf="$WORK/$label.elf"
  local log="$WORK/$label.runtime.log"
  local magic=""
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
}

compile_source positive-noopt disabled "$POSITIVE"
require_receipt positive-noopt '^\[rebracket-native-v2\] phase=single-disabled schema=1 optimize=0 functions=[0-9]+ transactions=0 authorizations=0 applications=0 refusals=0 same_region_applications=0 cross_block_applications=0 refusal_reason_mask=0 transaction_ontology_parameter_links=0 application_ontology_parameter_links=0 last_ontology_class_hash=0 operator_mask=0 last_function=-1 combined=0$'
run_elf_expect_zero positive-noopt

compile_source positive-cross optimized "$POSITIVE"
require_receipt positive-cross "^\\[rebracket-native-v2\\] phase=single-post-resolve schema=1 optimize=1 functions=[0-9]+ transactions=1 authorizations=1 applications=1 refusals=0 same_region_applications=0 cross_block_applications=1 refusal_reason_mask=0 transaction_ontology_parameter_links=1 application_ontology_parameter_links=1 last_ontology_class_hash=$ONTOLOGY_CLASS_HASH operator_mask=1 last_function=[0-9]+ combined=15$"
run_elf_expect_zero positive-cross

compile_source positive-imported-cross optimized "$IMPORTED_POSITIVE"
require_text 'module_native_driver: -O selects finalized full IR cleanup path' "$WORK/positive-imported-cross.compile.log"
require_receipt positive-imported-cross "^\\[rebracket-native-v2\\] phase=merged-post-finalize schema=1 optimize=1 functions=[0-9]+ transactions=1 authorizations=1 applications=1 refusals=0 same_region_applications=0 cross_block_applications=1 refusal_reason_mask=0 transaction_ontology_parameter_links=1 application_ontology_parameter_links=1 last_ontology_class_hash=$ONTOLOGY_CLASS_HASH operator_mask=1 last_function=[0-9]+ combined=15$"
run_elf_expect_zero positive-imported-cross

compile_source nondominating-control optimized "$NONDOMINATING"
require_receipt nondominating-control '^\[rebracket-native-v2\] phase=single-post-resolve schema=1 optimize=1 functions=[0-9]+ transactions=0 authorizations=0 applications=0 refusals=0 same_region_applications=0 cross_block_applications=0 refusal_reason_mask=0 transaction_ontology_parameter_links=0 application_ontology_parameter_links=0 last_ontology_class_hash=0 operator_mask=0 last_function=-1 combined=0$'
run_elf_expect_zero nondominating-control

compile_source loop-refusal optimized "$LOOP_REFUSAL"
require_receipt loop-refusal "^\\[rebracket-native-v2\\] phase=single-post-resolve schema=1 optimize=1 functions=[0-9]+ transactions=1 authorizations=0 applications=0 refusals=1 same_region_applications=0 cross_block_applications=0 refusal_reason_mask=524288 transaction_ontology_parameter_links=1 application_ontology_parameter_links=0 last_ontology_class_hash=$ONTOLOGY_CLASS_HASH operator_mask=0 last_function=-1 combined=0$"
run_elf_expect_zero loop-refusal

echo "[rebracket-source-ir] PASS: source_preflight=4 ontology_rejection=1 ontology_class_hash=$ONTOLOGY_CLASS_HASH positive_disabled=1 positive_cross=1 imported_cross=1 nondominating_candidate=0 loop_refusal_reason=19 ontology_cannot_authorize_loop=1 runtime_parity=5 compiler_sha256=$compiler_sha256 source_sha=$source_sha merge_ready=1"
