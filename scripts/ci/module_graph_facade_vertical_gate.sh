#!/usr/bin/env bash
# Behavioral acceptance gate for the canonical ModuleGraph facade vertical.
#
# Scope:
#   integrated root -> public facade -> leaf closure -> checker -> lowering -> ELF stdout 42
#   named public facade forwarding with selective fail-closed visibility
#   missing-package closure refusal with no observed lowering markers
#   #854 contextual lookup classification while preserving real privacy errors
#   optional #991 shadow closure as a causal oracle, never as a gate dependency

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="$ROOT_DIR/bin/madaros"
RAW_MADAROS="${SOUNIO_MODULE_GRAPH_FACADE_RAW_BIN:-${SOUC_BIN:-}}"
EXPECTED_RAW_SHA256="${SOUNIO_MODULE_GRAPH_FACADE_EXPECTED_RAW_SHA256:-}"
KEEP_WORK="${SOUNIO_MODULE_GRAPH_FACADE_KEEP:-0}"
TIMEOUT_SECONDS="${SOUNIO_MODULE_GRAPH_FACADE_TIMEOUT_SECONDS:-360}"
ISSUE_991_ROOT="${SOUNIO_MODULE_GRAPH_FACADE_ISSUE_991_ROOT:-}"
ISSUE_991_EXPECTED_HEAD="${SOUNIO_MODULE_GRAPH_FACADE_ISSUE_991_EXPECTED_HEAD:-}"

fail() {
  printf 'MODULE_GRAPH_FACADE_VERTICAL_FAIL reason=%s\n' "$1" >&2
  exit 1
}

blocked() {
  local reason="$1"
  local stage="$2"
  local issues="${3:-901,921,842}"
  trap - EXIT
  printf 'MODULE_GRAPH_FACADE_VERTICAL_BLOCKED blocker=BLK-20260719-MODULE-GRAPH-FACADE issue=%s stage=%s reason=%s work=%s\n' \
    "$issues" "$stage" "$reason" "$WORK" >&2
  exit 1
}

if [[ -n "${SOUNIO_MODULE_GRAPH_FACADE_DIR:-}" ]]; then
  WORK="$SOUNIO_MODULE_GRAPH_FACADE_DIR"
  [[ ! -e "$WORK" ]] || fail work_directory_already_exists
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-module-graph-facade.XXXXXX")"
fi

if [[ "$KEEP_WORK" != 1 ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

is_fatal_log() {
  grep -Eiq 'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' "$1"
}

has_forbidden_fallback() {
  grep -Eq 'native_prebundle:|falling back to full IR path' "$1"
}

has_legacy_compact_path() {
  grep -Eq 'compact modular IR table path|legacy compact IR differential enabled' "$1"
}

has_post_surface_work() {
  grep -Eq 'run_check_mode: about to check|run_check_mode: verdict=|typecheck:|imported_compile:|module_frontend_full_ir: lower_|lower_array:|Merged IR:|Compilation successful!' "$1"
}

closure_is_structurally_complete() {
  local report="$1"

  is_fatal_log "$report" && return 1
  [[ "$(grep -Fxc 'SOUNIO_BOUNDARY_CLOSURE_V1' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^status\t' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -Fxc $'status\tcomplete' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^saturated\t' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -Fxc $'saturated\tfalse' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^parse_failed\t' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -Fxc $'parse_failed\tfalse' "$report" || true)" -eq 1 ]] || return 1
  if grep -q $'^unresolved\t' "$report"; then
    return 1
  fi
  if grep -q $'^ambiguous\t' "$report"; then
    return 1
  fi
  return 0
}

closure_is_complete() {
  local report="$1"

  closure_is_structurally_complete "$report" || return 1
  [[ "$(grep -c $'^surface_status\t' "$report" || true)" -eq 1 ]] || return 1
  grep -Fxq $'surface_status\tvalid' "$report" || return 1
  ! grep -q $'^surface_error\t' "$report"
}

closure_is_missing_package() {
  local report="$1"
  local source="$2"
  local missing_source="${source%_package.sio}.sio"

  is_fatal_log "$report" && return 1
  [[ "$(grep -Fxc 'SOUNIO_BOUNDARY_CLOSURE_V1' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^status\t' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -Fxc $'status\tincomplete' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^saturated\t' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -Fxc $'saturated\tfalse' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^parse_failed\t' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -Fxc $'parse_failed\tfalse' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^surface_status\t' "$report" || true)" -eq 1 ]] || return 1
  grep -Fxq $'surface_status\tnot_evaluated' "$report" || return 1
  grep -Fxq $'node\t'"$source" "$report" || return 1
  grep -Fxq $'edge\t'"$source"$'\t'"$missing_source" "$report" || return 1
  grep -Fxq $'unresolved\t'"$source"$'\tpackage_import_missing' "$report" || return 1
  grep -Fxq $'logical_node\t0\tpackage_import_missing_package' "$report" || return 1
  grep -Fxq $'edge_identity\t0\t-1\tpackage_import_missing' "$report" || return 1
  [[ "$(grep -c $'^node\t' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^edge\t' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^logical_node\t' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^edge_identity\t' "$report" || true)" -eq 1 ]] || return 1
  [[ "$(grep -c $'^unresolved\t' "$report" || true)" -eq 1 ]] || return 1
  ! grep -q $'^ambiguous\t' "$report" || return 1
  ! grep -q $'^surface_error\t' "$report" || return 1
  return 0
}

require_closure_node() {
  local report="$1"
  local path="$2"
  grep -Fxq $'node\t'"$path" "$report"
}

require_closure_edge() {
  local report="$1"
  local from="$2"
  local to="$3"
  grep -Fxq $'edge\t'"$from"$'\t'"$to" "$report"
}

require_closure_cardinality() {
  local report="$1"
  local expected_nodes="$2"
  local expected_edges="$3"

  [[ "$(grep -c $'^node\t' "$report" || true)" -eq "$expected_nodes" ]] || return 1
  [[ "$(grep -c $'^edge\t' "$report" || true)" -eq "$expected_edges" ]] || return 1
}

require_closure_identity_cardinality() {
  local report="$1"
  local expected_nodes="$2"
  local expected_edges="$3"

  [[ "$(grep -c $'^logical_node\t' "$report" || true)" -eq "$expected_nodes" ]] || return 1
  [[ "$(grep -c $'^edge_identity\t' "$report" || true)" -eq "$expected_edges" ]] || return 1
}

require_surface_error() {
  local report="$1"
  local caller="$2"
  local requested_import="$3"
  local symbol="$4"

  [[ "$(grep -c $'^surface_status\t' "$report" || true)" -eq 1 ]] || return 1
  grep -Fxq $'surface_status\tinvalid' "$report" || return 1
  [[ "$(grep -c $'^surface_error\t' "$report" || true)" -eq 1 ]] || return 1
  grep -Fxq $'surface_error\t'"$caller"$'\t'"$requested_import"$'\t'"$symbol" "$report"
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual

  [[ -f "$path" ]] || fail "${label}_missing"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail "${label}_sha256_mismatch"
}

semantic_symbol_rejection() {
  local rc="$1"
  local log="$2"
  local source="$3"
  local symbol="$4"
  local requested_import="$5"
  local compiler_diag_count
  local plain_error
  local expected

  [[ "$rc" -eq 1 ]] || return 1
  is_fatal_log "$log" && return 1
  has_forbidden_fallback "$log" && return 1
  has_post_surface_work "$log" && return 1
  grep -Fq 'AST closure incomplete' "$log" && return 1
  grep -Fq 'check: OK' "$log" && return 1
  compiler_diag_count="$(grep -oE 'error\[E[0-9]{3}\]' "$log" | wc -l | tr -d ' ' || true)"
  [[ "$compiler_diag_count" -eq 1 ]] || return 1
  expected="error[E137]: imported name \`$symbol\` is not exported by \`$requested_import\` for caller $source"
  [[ "$(grep -Fxc "$expected" "$log" || true)" -eq 1 ]] || return 1

  while IFS= read -r plain_error; do
    case "$plain_error" in
      'error: visibility preflight failed') ;;
      'error: madaros build: compiler produced no ELF at '*) ;;
      *) return 1 ;;
    esac
  done < <(grep -E '^error:' "$log" || true)
}

assert_no_artifact() {
  [[ ! -e "$1" ]]
}

run_classifier_self_test() {
  local source="$WORK/selftest.sio"
  local valid_log="$WORK/selftest.valid.log"
  local fatal_log="$WORK/selftest.fatal.log"
  local mixed_log="$WORK/selftest.mixed.log"
  local legacy_mixed_log="$WORK/selftest.legacy-mixed.log"
  local post_surface_log="$WORK/selftest.post-surface.log"
  local misleading_log="$WORK/selftest.misleading.log"
  local fallback_log="$WORK/selftest.fallback.log"
  local compact_log="$WORK/selftest.compact.log"
  local legacy_compact_log="$WORK/selftest.legacy-compact.log"
  local target_log="$WORK/selftest.target.log"
  local complete="$WORK/selftest.complete.tsv"
  local surface_invalid="$WORK/selftest.surface-invalid.tsv"
  local incomplete="$WORK/selftest.incomplete.tsv"
  local conflicting="$WORK/selftest.conflicting.tsv"
  local missing_source="$WORK/package_import_missing_package.sio"
  local missing_report="$WORK/selftest.missing-package.tsv"
  local mixed_missing_report="$WORK/selftest.mixed-missing-package.tsv"

  printf '%s\n' \
    'fn main() -> i64 {' \
    '    synthetic_value()' \
    '}' >"$source"
  printf 'error[E137]: imported name `synthetic_value` is not exported by `facade` for caller %s\n' \
    "$source" >"$valid_log"
  semantic_symbol_rejection 1 "$valid_log" "$source" synthetic_value facade || fail selftest_valid_semantic_rejection
  if semantic_symbol_rejection 139 "$valid_log" "$source" synthetic_value facade; then
    fail selftest_rc139_accepted
  fi

  cp "$valid_log" "$mixed_log"
  printf '%s\n' 'error[E137]: imported name `other` is not exported by `facade` for caller /tmp/other.sio' >>"$mixed_log"
  if semantic_symbol_rejection 1 "$mixed_log" "$source" synthetic_value facade; then
    fail selftest_unrelated_diagnostic_accepted
  fi

  cp "$valid_log" "$legacy_mixed_log"
  printf '%s\n' 'error: unknown identifier `synthetic_value` at 0..15' >>"$legacy_mixed_log"
  if semantic_symbol_rejection 1 "$legacy_mixed_log" "$source" synthetic_value facade; then
    fail selftest_legacy_diagnostic_accepted_with_causal_e137
  fi

  cp "$valid_log" "$post_surface_log"
  printf '%s\n' 'module_frontend_full_ir: lower_node module_id=0 logical= physical=/tmp/selftest.sio' >>"$post_surface_log"
  if semantic_symbol_rejection 1 "$post_surface_log" "$source" synthetic_value facade; then
    fail selftest_post_surface_lowering_accepted
  fi

  printf '%s\n' \
    'error[E999]: incompatible operands output=/tmp/private.elf' \
    'error: madaros build: compiler produced no ELF at /tmp/private.elf' >"$misleading_log"
  if semantic_symbol_rejection 1 "$misleading_log" "$source" synthetic_value facade; then
    fail selftest_misleading_path_diagnostic_accepted
  fi

  cp "$valid_log" "$fatal_log"
  printf '%s\n' 'fatal: synthetic compiler abort' >>"$fatal_log"
  if semantic_symbol_rejection 1 "$fatal_log" "$source" synthetic_value facade; then
    fail selftest_fatal_semantic_rejection_accepted
  fi

  printf '%s\n' 'module_native_driver: compact IR ELF write failed; falling back to full IR path' >"$fallback_log"
  has_forbidden_fallback "$fallback_log" || fail selftest_fallback_not_detected
  printf '%s\n' 'module_native_driver: imported source uses compact modular IR table path' >"$compact_log"
  has_legacy_compact_path "$compact_log" || fail selftest_legacy_compact_not_detected
  printf '%s\n' 'module_native_driver: legacy compact IR differential enabled' >"$legacy_compact_log"
  has_legacy_compact_path "$legacy_compact_log" || fail selftest_legacy_compact_differential_not_detected
  printf '%s\n' 'Native dispatch: source=fallback fallback=unresolved_default_x86_64_linux' >"$target_log"
  if has_forbidden_fallback "$target_log"; then
    fail selftest_target_selection_misclassified
  fi

  printf '%s\n' \
    'SOUNIO_BOUNDARY_CLOSURE_V1' \
    $'status\tcomplete' \
    $'surface_status\tvalid' \
    $'capacity\t256' \
    $'saturated\tfalse' \
    $'parse_failed\tfalse' \
    $'node\t/tmp/root.sio' \
    $'logical_node\t0\troot' >"$complete"
  closure_is_complete "$complete" || fail selftest_complete_closure_rejected
  require_closure_cardinality "$complete" 1 0 || fail selftest_logical_node_counted_as_physical_node
  require_closure_identity_cardinality "$complete" 1 0 || fail selftest_logical_node_identity_rejected
  cp "$complete" "$surface_invalid"
  sed -i $'s/^surface_status\tvalid$/surface_status\tinvalid/' "$surface_invalid"
  printf '%s\n' $'surface_error\t/tmp/root.sio\tfacade\tsynthetic_value' >>"$surface_invalid"
  closure_is_structurally_complete "$surface_invalid" || fail selftest_surface_invalid_structure_rejected
  if closure_is_complete "$surface_invalid"; then
    fail selftest_surface_invalid_accepted_as_clean
  fi
  require_surface_error "$surface_invalid" /tmp/root.sio facade synthetic_value || fail selftest_surface_error_receipt_rejected
  cp "$complete" "$conflicting"
  printf '%s\n' $'status\tincomplete' >>"$conflicting"
  if closure_is_complete "$conflicting"; then
    fail selftest_conflicting_closure_state_accepted
  fi
  printf '%s\n' \
    'SOUNIO_BOUNDARY_CLOSURE_V1' \
    $'status\tincomplete' \
    $'surface_status\tnot_evaluated' \
    $'capacity\t256' \
    $'saturated\tfalse' \
    $'parse_failed\tfalse' \
    $'node\t/tmp/root.sio' \
    $'unresolved\t/tmp/root.sio\tmissing' >"$incomplete"
  if closure_is_complete "$incomplete"; then
    fail selftest_incomplete_closure_accepted
  fi

  printf '%s\n' 'use package_import_missing::*' >"$missing_source"
  printf '%s\n' \
    'SOUNIO_BOUNDARY_CLOSURE_V1' \
    $'status\tincomplete' \
    $'surface_status\tnot_evaluated' \
    $'capacity\t256' \
    $'saturated\tfalse' \
    $'parse_failed\tfalse' \
    $'node\t'"$missing_source" \
    $'logical_node\t0\tpackage_import_missing_package' \
    $'edge\t'"$missing_source"$'\t'"${missing_source%_package.sio}.sio" \
    $'edge_identity\t0\t-1\tpackage_import_missing' \
    $'unresolved\t'"$missing_source"$'\tpackage_import_missing' >"$missing_report"
  closure_is_missing_package "$missing_report" "$missing_source" || fail selftest_exact_missing_package_rejected
  cp "$missing_report" "$mixed_missing_report"
  printf '%s\n' $'ambiguous\t'"$missing_source"$'\tpackage_import_missing' >>"$mixed_missing_report"
  if closure_is_missing_package "$mixed_missing_report" "$missing_source"; then
    fail selftest_mixed_missing_package_accepted
  fi

  assert_no_artifact "$WORK/selftest.absent.elf" || fail selftest_absent_artifact_rejected
  printf '%s' ELF >"$WORK/selftest.present.elf"
  if assert_no_artifact "$WORK/selftest.present.elf"; then
    fail selftest_present_artifact_accepted
  fi
  printf 'MODULE_GRAPH_FACADE_VERTICAL_SELFTEST_PASS semantic_rc1=accepted rc139=rejected unrelated_diagnostic=rejected legacy_mixed_diagnostic=rejected post_surface_lowering=rejected misleading_path=rejected fatal=rejected fallback=detected legacy_compact=detected legacy_compact_differential=detected target_fallback=ignored closure_incomplete=rejected closure_conflict=rejected surface_invalid=separate missing_package=exact artifact_presence=detected\n'
}

run_classifier_self_test
if [[ "${SOUNIO_MODULE_GRAPH_FACADE_SELF_TEST_ONLY:-0}" == 1 ]]; then
  exit 0
fi

[[ "$(uname -s 2>/dev/null || true)" == Linux ]] || fail linux_required
case "$(uname -m 2>/dev/null || true)" in
  x86_64|amd64) ;;
  *) fail x86_64_required ;;
esac

[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail invalid_timeout_seconds
[[ -x "$WRAPPER" ]] || fail madaros_wrapper_missing
[[ -n "$RAW_MADAROS" ]] || fail explicit_raw_or_souc_bin_required
[[ -x "$RAW_MADAROS" ]] || fail explicit_raw_missing_or_not_executable
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
[[ "$EXPECTED_RAW_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail expected_raw_sha256_required
[[ "$(od -An -tx1 -N4 "$RAW_MADAROS" | tr -d ' \n')" == 7f454c46 ]] || fail raw_compiler_must_be_elf
RAW_SHA256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$RAW_SHA256" == "$EXPECTED_RAW_SHA256" ]] || fail raw_compiler_sha256_mismatch

set +e
"$RAW_MADAROS" --version >"$WORK/raw-version.log" 2>&1
RAW_VERSION_RC=$?
set -e
[[ "$RAW_VERSION_RC" -eq 0 ]] || fail raw_compiler_identity_failed
grep -Fq 'Madaros v' "$WORK/raw-version.log" || fail raw_compiler_identity_missing

set +e
timeout "$TIMEOUT_SECONDS" "$RAW_MADAROS" --module-path-shape-self-test >"$WORK/module-path-shape-self-test.log" 2>&1
PATH_SHAPE_RC=$?
set -e
[[ "$PATH_SHAPE_RC" -eq 0 ]] || blocked "module_path_shape_self_test_rc_${PATH_SHAPE_RC}" closure_identity
grep -Fxq 'module-path-shape-self-test: OK' "$WORK/module-path-shape-self-test.log" || \
  blocked module_path_shape_self_test_receipt_missing closure_identity
printf 'MODULE_GRAPH_PATH_SHAPE_PASS exact_arity=true head_bounds=1..128 malformed=fail_closed\n'

run_closure() {
  local label="$1"
  local cwd="$2"
  local stdlib="$3"
  local source="$4"
  local log="$WORK/$label.closure.tsv"

  set +e
  (
    cd "$cwd"
    SOUNIO_STDLIB_PATH="$stdlib" timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
      "$RAW_MADAROS" --science-boundary-closure "$source"
  ) >"$log" 2>&1
  CASE_RC=$?
  set -e
  CASE_LOG="$log"
}

run_check() {
  local label="$1"
  local cwd="$2"
  local stdlib="$3"
  local source="$4"
  local log="$WORK/$label.check.log"

  set +e
  (
    cd "$cwd"
    MADAROS_RAW_BIN="$RAW_MADAROS" SOUNIO_STDLIB_PATH="$stdlib" \
      timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
      "$WRAPPER" --science-boundary off check "$source"
  ) >"$log" 2>&1
  CASE_RC=$?
  set -e
  CASE_LOG="$log"
}

run_build() {
  local label="$1"
  local source_cwd="$2"
  local stdlib="$3"
  local source="$4"
  local elf="$5"
  local log="$WORK/$label.build.log"
  local build_cwd="$WORK/$label.build-cwd"

  [[ -d "$source_cwd" ]] || fail "${label}_source_cwd_missing"
  rm -rf "$build_cwd"
  mkdir -p "$build_cwd"
  rm -f "$elf"
  set +e
  (
    cd "$build_cwd"
    MADAROS_RAW_BIN="$RAW_MADAROS" SOUNIO_STDLIB_PATH="$stdlib" SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1 SOUNIO_LEGACY_COMPACT_IR=0 \
      timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
      "$WRAPPER" --science-boundary off build "$source" -o "$elf"
  ) >"$log" 2>&1
  CASE_RC=$?
  set -e
  CASE_LOG="$log"
  CASE_BUILD_DIR="$build_cwd"
}

assert_check_success() {
  local label="$1"
  local rc="$2"
  local log="$3"
  local closure_report="$4"
  local expected_modules
  local normalized="$WORK/$label.check.normalized.log"

  [[ "$rc" -eq 0 ]] || {
    tail -n 100 "$log" >&2 || true
    blocked "${label}_checker_rc_${rc}" modular_checker
  }
  is_fatal_log "$log" && blocked "${label}_checker_fatal" modular_checker
  grep -Fq 'run_check_mode: verdict=0' "$log" || blocked "${label}_checker_verdict_missing" modular_checker
  grep -Fq 'check: OK' "$log" || blocked "${label}_checker_ok_missing" modular_checker
  if grep -Fq 'error[E' "$log" || grep -Eq '^error:' "$log"; then
    blocked "${label}_checker_diagnostic_on_success" modular_checker
  fi
  expected_modules="$(grep -c $'^node\t' "$closure_report" || true)"
  tr -d '\r\n' <"$log" >"$normalized"
  grep -Fq "run_check_mode: about to check ${expected_modules} modules" "$normalized" || blocked "${label}_checker_closure_count_mismatch" modular_checker
}

assert_build_success() {
  local label="$1"
  local rc="$2"
  local log="$3"
  local elf="$4"

  [[ "$rc" -eq 0 ]] || {
    tail -n 120 "$log" >&2 || true
    blocked "${label}_build_rc_${rc}" lowering
  }
  is_fatal_log "$log" && blocked "${label}_build_fatal" lowering
  has_forbidden_fallback "$log" && {
    tail -n 120 "$log" >&2 || true
    blocked "${label}_forbidden_fallback" lowering
  }
  has_legacy_compact_path "$log" && blocked "${label}_legacy_compact_path" lowering
  grep -Fq 'canonical AST closure full IR path' "$log" || blocked "${label}_canonical_full_ir_marker_missing" lowering
  grep -Fq 'Merged IR:' "$log" || blocked "${label}_merged_ir_missing" lowering
  grep -Fq 'Compilation successful!' "$log" || blocked "${label}_compile_success_marker_missing" lowering
  if grep -Fq 'error[E' "$log" || grep -Eq '^error:' "$log"; then
    blocked "${label}_compiler_diagnostic_on_build_success" lowering
  fi
  [[ -f "$elf" ]] || blocked "${label}_elf_missing" elf
  [[ "$(od -An -tx1 -N4 "$elf" | tr -d ' \n')" == 7f454c46 ]] || blocked "${label}_artifact_not_elf" elf
  if find "$CASE_BUILD_DIR" -type f -print -quit | grep -q .; then
    find "$CASE_BUILD_DIR" -type f -print >&2
    blocked "${label}_unexpected_build_cwd_artifact" elf
  fi
}

assert_full_ir_input_receipt_matches_closure() {
  local label="$1"
  local log="$2"
  local closure_report="$3"
  local expected_nodes="$WORK/$label.lowering.expected.nodes"
  local actual_nodes="$WORK/$label.lowering.actual.nodes"
  local expected_edges="$WORK/$label.lowering.expected.edges"
  local actual_edges="$WORK/$label.lowering.actual.edges"

  awk -F '\t' '
    $1 == "node" { physical[node_index] = $2; node_index = node_index + 1 }
    $1 == "logical_node" {
      printf "module_frontend_full_ir: lower_node module_id=%s logical=%s physical=%s\n", $2, $3, physical[$2]
    }
  ' "$closure_report" >"$expected_nodes"
  grep '^module_frontend_full_ir: lower_node ' "$log" >"$actual_nodes" || true

  awk -F '\t' '
    $1 == "edge_identity" {
      printf "module_frontend_full_ir: lower_edge caller_id=%s dependency_id=%s import=%s\n", $2, $3, $4
    }
  ' "$closure_report" >"$expected_edges"
  grep '^module_frontend_full_ir: lower_edge ' "$log" >"$actual_edges" || true

  [[ -s "$actual_nodes" ]] || blocked "${label}_full_ir_lower_node_trace_missing" lowering
  cmp -s "$expected_nodes" "$actual_nodes" || {
    diff -u "$expected_nodes" "$actual_nodes" >&2 || true
    blocked "${label}_full_ir_input_node_receipt_mismatch" lowering
  }
  cmp -s "$expected_edges" "$actual_edges" || {
    diff -u "$expected_edges" "$actual_edges" >&2 || true
    blocked "${label}_full_ir_input_edge_receipt_mismatch" lowering
  }
}

run_elf_exact_stdout() {
  local label="$1"
  local cwd="$2"
  local elf="$3"
  local expected="$4"
  local trailing_newline="${5:-0}"
  local stdout="$WORK/$label.stdout"
  local stderr="$WORK/$label.stderr"
  local expected_stdout="$WORK/$label.expected.stdout"

  chmod +x "$elf"
  set +e
  (
    cd "$cwd"
    timeout --signal=TERM --kill-after=5s 60 "$elf"
  ) >"$stdout" 2>"$stderr"
  RUNTIME_RC=$?
  set -e
  [[ "$RUNTIME_RC" -eq 0 ]] || {
    cat "$stdout" >&2 || true
    cat "$stderr" >&2 || true
    blocked "${label}_runtime_rc_${RUNTIME_RC}" runtime
  }
  is_fatal_log "$stderr" && blocked "${label}_runtime_fatal" runtime
  if [[ "$trailing_newline" == 1 ]]; then
    printf '%s\n' "$expected" >"$expected_stdout"
  else
    printf '%s' "$expected" >"$expected_stdout"
  fi
  cmp -s "$stdout" "$expected_stdout" || {
    cat "$stdout" >&2 || true
    blocked "${label}_stdout_mismatch" runtime
  }
}

ISSUE_991_STATE=not_run
ISSUE_991_HEAD=none
if [[ -n "$ISSUE_991_ROOT" ]]; then
  [[ -d "$ISSUE_991_ROOT" ]] || fail issue_991_root_missing
  ISSUE_991_ROOT="$(cd "$ISSUE_991_ROOT" && pwd)"
  [[ "$ISSUE_991_EXPECTED_HEAD" =~ ^[0-9a-f]{40,64}$ ]] || fail issue_991_expected_head_required
  ISSUE_991_HEAD="$(git -C "$ISSUE_991_ROOT" rev-parse HEAD 2>/dev/null || true)"
  [[ "$ISSUE_991_HEAD" == "$ISSUE_991_EXPECTED_HEAD" ]] || fail issue_991_head_mismatch
  [[ -z "$(git -C "$ISSUE_991_ROOT" status --porcelain 2>/dev/null)" ]] || fail issue_991_root_must_be_clean

  ISSUE_991_WITNESS="$ISSUE_991_ROOT/tests/native-v2/nominal_field_resolution_receipt_shadow_witness.sio"
  ISSUE_991_PROBE="$ISSUE_991_ROOT/self-hosted/check/nominal_field_resolution_receipt_shadow_probe.sio"
  ISSUE_991_MODULE_PARSE="$ISSUE_991_ROOT/self-hosted/compiler/module_parse.sio"
  ISSUE_991_WRONG_MODULE_PARSE="$ISSUE_991_ROOT/self-hosted/check/module_parse.sio"
  require_sha256 "$ISSUE_991_WITNESS" 9e7188f10893b796e01fa1576fd752f03a5139b7ee39f27b3de2e0efb8660c70 issue_991_witness
  require_sha256 "$ISSUE_991_PROBE" e7ef5cff6b754844f34d6afb10917a3f35bae2253f79d3b623c756faa7e42cf6 issue_991_probe
  [[ -f "$ISSUE_991_MODULE_PARSE" ]] || fail issue_991_module_parse_missing

  run_closure issue-991-oracle "$ISSUE_991_ROOT" "$ISSUE_991_ROOT/self-hosted" "$ISSUE_991_WITNESS"
  ISSUE_991_LOG="$CASE_LOG"
  if [[ "$CASE_RC" -ne 0 ]]; then
    ISSUE_991_STATE="closure_command_rc_${CASE_RC}"
  elif closure_is_complete "$ISSUE_991_LOG" &&
      require_closure_node "$ISSUE_991_LOG" "$ISSUE_991_WITNESS" &&
      require_closure_node "$ISSUE_991_LOG" "$ISSUE_991_PROBE" &&
      require_closure_node "$ISSUE_991_LOG" "$ISSUE_991_MODULE_PARSE" &&
      require_closure_edge "$ISSUE_991_LOG" "$ISSUE_991_WITNESS" "$ISSUE_991_PROBE" &&
      require_closure_edge "$ISSUE_991_LOG" "$ISSUE_991_PROBE" "$ISSUE_991_MODULE_PARSE" &&
      ! require_closure_node "$ISSUE_991_LOG" "$ISSUE_991_WRONG_MODULE_PARSE" &&
      ! require_closure_edge "$ISSUE_991_LOG" "$ISSUE_991_PROBE" "$ISSUE_991_WRONG_MODULE_PARSE"; then
    ISSUE_991_STATE=closure_ready
  elif require_closure_edge "$ISSUE_991_LOG" "$ISSUE_991_PROBE" "$ISSUE_991_WRONG_MODULE_PARSE" &&
      grep -Fxq $'unresolved\t'"$ISSUE_991_PROBE"$'\tmodule_parse' "$ISSUE_991_LOG"; then
    ISSUE_991_STATE=logical_physical_module_parse_mismatch
  else
    ISSUE_991_STATE=closure_incomplete_other
  fi
  printf 'MODULE_GRAPH_FACADE_ISSUE_991_ORACLE state=%s head=%s witness_sha256=9e7188f10893b796e01fa1576fd752f03a5139b7ee39f27b3de2e0efb8660c70 probe_sha256=e7ef5cff6b754844f34d6afb10917a3f35bae2253f79d3b623c756faa7e42cf6 gate_dependency=false\n' \
    "$ISSUE_991_STATE" "$ISSUE_991_HEAD"
fi

IDENTITY_DIR="$ROOT_DIR/tests/compiler/module_declaration_identity"
IDENTITY_CONSUMER="$IDENTITY_DIR/alias_consumer.sio"
IDENTITY_DECLARED="$IDENTITY_DIR/left/mod.sio"
require_sha256 "$IDENTITY_CONSUMER" 42ee4f4cebf9fbde5d80f53e4c79e4e9029b73a5bf4d5c40dceedd4c0011b0ec identity_alias_consumer
require_sha256 "$IDENTITY_DECLARED" 1ab4de8250dc5363cec1646788392a7603e3ac8d350e96a62b5f379b0cf033cb identity_declared_module
run_closure declared-identity "$ROOT_DIR" "$ROOT_DIR/stdlib" "$IDENTITY_CONSUMER"
[[ "$CASE_RC" -eq 0 ]] || blocked "declared_identity_closure_rc_${CASE_RC}" closure_identity
closure_is_complete "$CASE_LOG" || blocked declared_identity_closure_incomplete closure_identity
require_closure_cardinality "$CASE_LOG" 2 1 || blocked declared_identity_closure_cardinality closure_identity
require_closure_identity_cardinality "$CASE_LOG" 2 1 || blocked declared_identity_receipt_cardinality closure_identity
require_closure_node "$CASE_LOG" "$IDENTITY_CONSUMER" || blocked declared_identity_consumer_node_missing closure_identity
require_closure_node "$CASE_LOG" "$IDENTITY_DECLARED" || blocked declared_identity_dependency_node_missing closure_identity
require_closure_edge "$CASE_LOG" "$IDENTITY_CONSUMER" "$IDENTITY_DECLARED" || blocked declared_identity_physical_edge_missing closure_identity
grep -Fxq $'logical_node\t0\talias_consumer' "$CASE_LOG" || blocked declared_identity_root_mismatch closure_identity
grep -Fxq $'logical_node\t1\tcollision::left' "$CASE_LOG" || blocked declared_identity_dependency_mismatch closure_identity
grep -Fxq $'edge_identity\t0\t1\tleft::mod' "$CASE_LOG" || blocked declared_identity_authored_edge_mismatch closure_identity
printf 'MODULE_GRAPH_DECLARED_IDENTITY_PASS node_identity=collision::left authored_import=left::mod distinctions=preserved\n'

CAPACITY_DIR="$WORK/import-capacity"
mkdir -p "$CAPACITY_DIR"
printf '%s\n' 'pub fn capacity_value() -> i64 { 1 }' >"$CAPACITY_DIR/dep.sio"
make_import_capacity_root() {
  local path="$1"
  local count="$2"
  local i=0
  : >"$path"
  while [[ "$i" -lt "$count" ]]; do
    printf '%s\n' 'use dep::*' >>"$path"
    i=$((i + 1))
  done
}

CAPACITY_256="$CAPACITY_DIR/imports_256.sio"
make_import_capacity_root "$CAPACITY_256" 256
run_closure imports-256 "$CAPACITY_DIR" "$ROOT_DIR/stdlib" "$CAPACITY_256"
[[ "$CASE_RC" -eq 0 ]] || blocked "imports_256_closure_rc_${CASE_RC}" closure_capacity
closure_is_complete "$CASE_LOG" || blocked imports_256_closure_incomplete closure_capacity
require_closure_cardinality "$CASE_LOG" 2 256 || blocked imports_256_closure_cardinality closure_capacity
require_closure_identity_cardinality "$CASE_LOG" 2 256 || blocked imports_256_identity_cardinality closure_capacity

CAPACITY_257="$CAPACITY_DIR/imports_257.sio"
make_import_capacity_root "$CAPACITY_257" 257
run_closure imports-257 "$CAPACITY_DIR" "$ROOT_DIR/stdlib" "$CAPACITY_257"
[[ "$CASE_RC" -eq 0 ]] || blocked "imports_257_closure_rc_${CASE_RC}" closure_capacity
grep -Fxq $'status\tincomplete' "$CASE_LOG" || blocked imports_257_status_not_incomplete closure_capacity
grep -Fxq $'surface_status\tnot_evaluated' "$CASE_LOG" || blocked imports_257_surface_was_claimed closure_capacity
grep -Fxq $'saturated\ttrue' "$CASE_LOG" || blocked imports_257_not_saturated closure_capacity
require_closure_cardinality "$CASE_LOG" 2 256 || blocked imports_257_closure_cardinality closure_capacity
require_closure_identity_cardinality "$CASE_LOG" 2 256 || blocked imports_257_identity_cardinality closure_capacity
if grep -q $'^unresolved\t\|^ambiguous\t\|^surface_error\t' "$CASE_LOG"; then
  blocked imports_257_wrong_failure_class closure_capacity
fi
printf 'MODULE_GRAPH_IMPORT_CAPACITY_PASS imports_256=complete imports_257=saturated fail_closed=true\n'

VERTICAL_DIR="$ROOT_DIR/tests/compiler/module_graph_facade_vertical"
VERTICAL_MAIN="$VERTICAL_DIR/main.sio"
VERTICAL_FACADE="$VERTICAL_DIR/facade.sio"
VERTICAL_LEAF="$VERTICAL_DIR/leaf.sio"
VERTICAL_MAIN_SHA256=2b66fa0df08cf71510ab294ba51bb82ea91af55371c29c67784d1814dbc51fab
require_sha256 "$VERTICAL_MAIN" "$VERTICAL_MAIN_SHA256" vertical_main
require_sha256 "$VERTICAL_FACADE" af35d468f4fd52a16aa386c363f903266e419a77999a65fdeb48457e7c151e5d vertical_facade
require_sha256 "$VERTICAL_LEAF" d1f764b370bdb2895e5338e77c4f8a80693cb7d1c72019138bfe8e6246e07447 vertical_leaf

run_closure vertical "$ROOT_DIR" "$ROOT_DIR/stdlib" "$VERTICAL_MAIN"
[[ "$CASE_RC" -eq 0 ]] || blocked "vertical_closure_command_rc_${CASE_RC}" closure
closure_is_structurally_complete "$CASE_LOG" || blocked vertical_closure_incomplete closure
closure_is_complete "$CASE_LOG" || blocked vertical_surface_receipt_not_valid closure
require_closure_cardinality "$CASE_LOG" 3 2 || blocked vertical_closure_cardinality closure
require_closure_identity_cardinality "$CASE_LOG" 3 2 || blocked vertical_closure_identity_cardinality closure
require_closure_node "$CASE_LOG" "$VERTICAL_MAIN" || blocked vertical_root_node_missing closure
require_closure_node "$CASE_LOG" "$VERTICAL_FACADE" || blocked vertical_facade_node_missing closure
require_closure_node "$CASE_LOG" "$VERTICAL_LEAF" || blocked vertical_leaf_node_missing closure
require_closure_edge "$CASE_LOG" "$VERTICAL_MAIN" "$VERTICAL_FACADE" || blocked vertical_root_facade_edge_missing closure
require_closure_edge "$CASE_LOG" "$VERTICAL_FACADE" "$VERTICAL_LEAF" || blocked vertical_facade_leaf_edge_missing closure
VERTICAL_CLOSURE="$CASE_LOG"

run_check vertical "$ROOT_DIR" "$ROOT_DIR/stdlib" "$VERTICAL_MAIN"
assert_check_success vertical "$CASE_RC" "$CASE_LOG" "$VERTICAL_CLOSURE"

VERTICAL_ELF="$WORK/vertical.elf"
run_build vertical "$ROOT_DIR" "$ROOT_DIR/stdlib" "$VERTICAL_MAIN" "$VERTICAL_ELF"
assert_build_success vertical "$CASE_RC" "$CASE_LOG" "$VERTICAL_ELF"
assert_full_ir_input_receipt_matches_closure vertical "$CASE_LOG" "$VERTICAL_CLOSURE"
run_elf_exact_stdout vertical "$ROOT_DIR" "$VERTICAL_ELF" 42 1
VERTICAL_ELF_SHA256="$(sha256sum "$VERTICAL_ELF" | awk '{print $1}')"
printf 'MODULE_GRAPH_FACADE_VERTICAL_SLICE_PASS closure_nodes=3 closure_edges=2 checker_modules=3 full_ir_input_nodes=3 full_ir_input_edges=2 lowering=executed lowering_identity_consumption=unproven elf_sha256=%s runtime_stdout=42_LF fallback=none\n' "$VERTICAL_ELF_SHA256"

FACADE_DIR="$ROOT_DIR/tests/compiler/pub_use_reexport"
PUBLIC_CONSUMER="$FACADE_DIR/public_consumer.sio"
PUBLIC_FACADE="$FACADE_DIR/public_facade.sio"
PUBLIC_LEAF="$FACADE_DIR/public_leaf.sio"
require_sha256 "$PUBLIC_CONSUMER" d28f7f48eb3395d664464966b180f7477e13b5614a0f68118cd7a1ce04ab2bf3 public_consumer
require_sha256 "$PUBLIC_FACADE" 9dd79bd6d9d0cf83321f1e338f4fbf71a4a233300e24e5f08978b9434953f179 public_facade
require_sha256 "$PUBLIC_LEAF" 3e1c28a38a848083be4b18809fdd74d5ce49496450874f46a51e574e8fac3a26 public_leaf
require_sha256 "$FACADE_DIR/missing_consumer.sio" aea7bef7b8e429e3bb8ee098d97ad9acbb6cf09d8616a37a6aa0ac2abfa8bcba missing_consumer
require_sha256 "$FACADE_DIR/not_reexported_consumer.sio" c654523071fca1b916555d7d0a68c810a4032a32863b5ac1065376ae8ea3641d not_reexported_consumer
require_sha256 "$FACADE_DIR/private_consumer.sio" 26d802cba4a3985673ba1aeab9ae0565aad7ea83709339040b1a36d224e5ae36 private_consumer
require_sha256 "$FACADE_DIR/private_facade.sio" 56ebc1da56c9e9b572c791e862d0876e0f7529ce3afa787b117e42c023c3846e private_facade
require_sha256 "$FACADE_DIR/private_leaf.sio" 527679a259a570018d18333cc99f62f81cd5cda136e96c33b51a32467117527f private_leaf

run_closure public-facade "$ROOT_DIR" "$ROOT_DIR/stdlib" "$PUBLIC_CONSUMER"
[[ "$CASE_RC" -eq 0 ]] || blocked "public_facade_closure_rc_${CASE_RC}" facade_closure
closure_is_structurally_complete "$CASE_LOG" || blocked public_facade_closure_incomplete facade_closure
closure_is_complete "$CASE_LOG" || blocked public_facade_surface_receipt_not_valid facade_closure
require_closure_cardinality "$CASE_LOG" 3 2 || blocked public_facade_closure_cardinality facade_closure
require_closure_identity_cardinality "$CASE_LOG" 3 2 || blocked public_facade_closure_identity_cardinality facade_closure
require_closure_node "$CASE_LOG" "$PUBLIC_CONSUMER" || blocked public_consumer_node_missing facade_closure
require_closure_node "$CASE_LOG" "$PUBLIC_FACADE" || blocked public_facade_node_missing facade_closure
require_closure_node "$CASE_LOG" "$PUBLIC_LEAF" || blocked public_leaf_node_missing facade_closure
require_closure_edge "$CASE_LOG" "$PUBLIC_CONSUMER" "$PUBLIC_FACADE" || blocked public_consumer_facade_edge_missing facade_closure
require_closure_edge "$CASE_LOG" "$PUBLIC_FACADE" "$PUBLIC_LEAF" || blocked public_facade_leaf_edge_missing facade_closure
PUBLIC_CLOSURE="$CASE_LOG"

run_check public-facade "$ROOT_DIR" "$ROOT_DIR/stdlib" "$PUBLIC_CONSUMER"
assert_check_success public_facade "$CASE_RC" "$CASE_LOG" "$PUBLIC_CLOSURE"
printf 'MODULE_GRAPH_FACADE_PUBLIC_PREFLIGHT_PASS closure_nodes=3 closure_edges=2 checker_modules=3\n'

PUBLIC_ELF="$WORK/public-facade.elf"
run_build public-facade "$ROOT_DIR" "$ROOT_DIR/stdlib" "$PUBLIC_CONSUMER" "$PUBLIC_ELF"
assert_build_success public_facade "$CASE_RC" "$CASE_LOG" "$PUBLIC_ELF"
assert_full_ir_input_receipt_matches_closure public_facade "$CASE_LOG" "$PUBLIC_CLOSURE"
run_elf_exact_stdout public-facade "$ROOT_DIR" "$PUBLIC_ELF" 'PASS pub_use_named_function_reexport' 1

run_facade_negative() {
  local label="$1"
  local source="$2"
  local symbol="$3"
  local facade="$4"
  local leaf="$5"
  local requested_import="$6"
  local elf="$WORK/$label.elf"

  [[ -f "$source" ]] || fail "${label}_fixture_missing"
  run_closure "$label" "$ROOT_DIR" "$ROOT_DIR/stdlib" "$source"
  [[ "$CASE_RC" -eq 0 ]] || blocked "${label}_closure_rc_${CASE_RC}" facade_negative_closure
  closure_is_structurally_complete "$CASE_LOG" || blocked "${label}_closure_incomplete" facade_negative_closure
  require_closure_cardinality "$CASE_LOG" 3 2 || blocked "${label}_closure_cardinality" facade_negative_closure
  require_closure_identity_cardinality "$CASE_LOG" 3 2 || blocked "${label}_closure_identity_cardinality" facade_negative_closure
  require_closure_node "$CASE_LOG" "$source" || blocked "${label}_consumer_node_missing" facade_negative_closure
  require_closure_node "$CASE_LOG" "$facade" || blocked "${label}_facade_node_missing" facade_negative_closure
  require_closure_node "$CASE_LOG" "$leaf" || blocked "${label}_leaf_node_missing" facade_negative_closure
  require_closure_edge "$CASE_LOG" "$source" "$facade" || blocked "${label}_consumer_facade_edge_missing" facade_negative_closure
  require_closure_edge "$CASE_LOG" "$facade" "$leaf" || blocked "${label}_facade_leaf_edge_missing" facade_negative_closure
  require_surface_error "$CASE_LOG" "$source" "$requested_import" "$symbol" || blocked "${label}_surface_error_not_exact" facade_visibility

  run_check "$label" "$ROOT_DIR" "$ROOT_DIR/stdlib" "$source"
  semantic_symbol_rejection "$CASE_RC" "$CASE_LOG" "$source" "$symbol" "$requested_import" || {
    cat "$CASE_LOG" >&2 || true
    blocked "${label}_check_not_semantic_rc1" facade_visibility
  }

  run_build "$label" "$ROOT_DIR" "$ROOT_DIR/stdlib" "$source" "$elf"
  semantic_symbol_rejection "$CASE_RC" "$CASE_LOG" "$source" "$symbol" "$requested_import" || {
    cat "$CASE_LOG" >&2 || true
    blocked "${label}_build_not_semantic_rc1" facade_visibility
  }
  assert_no_artifact "$elf" || blocked "${label}_emitted_elf" facade_visibility
  if find "$CASE_BUILD_DIR" -type f -print -quit | grep -q .; then
    find "$CASE_BUILD_DIR" -type f -print >&2
    blocked "${label}_unexpected_build_cwd_artifact" facade_visibility
  fi
  printf 'MODULE_GRAPH_FACADE_VERTICAL_NEGATIVE_PASS case=%s symbol=%s check_rc=1 build_rc=1 elf=absent\n' "$label" "$symbol"
}

run_facade_negative missing "$FACADE_DIR/missing_consumer.sio" missing_route_value "$PUBLIC_FACADE" "$PUBLIC_LEAF" public_facade
run_facade_negative not_reexported "$FACADE_DIR/not_reexported_consumer.sio" not_reexported_value "$PUBLIC_FACADE" "$PUBLIC_LEAF" public_facade
run_facade_negative private "$FACADE_DIR/private_consumer.sio" private_route_value "$FACADE_DIR/private_facade.sio" "$FACADE_DIR/private_leaf.sio" private_facade

MISSING_PACKAGE="$ROOT_DIR/tests/packages/package_import_missing_package.sio"
require_sha256 "$MISSING_PACKAGE" b95334c704dfa4aba033a08de743742e9560c56714b4fe9737a62dda16230df4 missing_package_fixture
run_closure missing-package "$ROOT_DIR" "$ROOT_DIR/stdlib" "$MISSING_PACKAGE"
[[ "$CASE_RC" -eq 0 ]] || blocked "missing_package_closure_rc_${CASE_RC}" missing_package
closure_is_missing_package "$CASE_LOG" "$MISSING_PACKAGE" || {
  cat "$CASE_LOG" >&2 || true
  blocked missing_package_closure_not_exact missing_package
}

MISSING_PACKAGE_ELF="$WORK/missing-package.elf"
run_build missing-package "$ROOT_DIR" "$ROOT_DIR/stdlib" "$MISSING_PACKAGE" "$MISSING_PACKAGE_ELF"
[[ "$CASE_RC" -eq 1 ]] || {
  cat "$CASE_LOG" >&2 || true
  blocked "missing_package_build_rc_${CASE_RC}" missing_package
}
is_fatal_log "$CASE_LOG" && blocked missing_package_fatal missing_package
has_forbidden_fallback "$CASE_LOG" && blocked missing_package_fallback missing_package
grep -Fq 'run_check_mode: AST closure incomplete' "$CASE_LOG" || blocked missing_package_fail_closed_marker_missing missing_package
if has_post_surface_work "$CASE_LOG"; then
  blocked missing_package_lowering_markers_present missing_package
fi
assert_no_artifact "$MISSING_PACKAGE_ELF" || blocked missing_package_emitted_elf missing_package
if find "$CASE_BUILD_DIR" -type f -print -quit | grep -q .; then
  find "$CASE_BUILD_DIR" -type f -print >&2
  blocked missing_package_unexpected_build_cwd_artifact missing_package
fi

VISIBILITY_LOG="$WORK/visibility-854.log"
set +e
MADAROS_RAW_BIN="$RAW_MADAROS" \
SOUNIO_MADAROS_VISIBILITY_CONTEXT_BIN="$WRAPPER" \
SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=classify \
SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
  bash "$ROOT_DIR/scripts/ci/madaros_visibility_context_gate.sh" >"$VISIBILITY_LOG" 2>&1
VISIBILITY_RC=$?
set -e
[[ "$VISIBILITY_RC" -eq 0 ]] || {
  tail -n 140 "$VISIBILITY_LOG" >&2 || true
  blocked "visibility_854_rc_${VISIBILITY_RC}" visibility_854 854
}
is_fatal_log "$VISIBILITY_LOG" && blocked visibility_854_fatal visibility_854 854
has_forbidden_fallback "$VISIBILITY_LOG" && blocked visibility_854_fallback visibility_854 854
grep -Fq 'true_private_fn=E175 true_private_struct=E176 true_private_enum=E177' "$VISIBILITY_LOG" || blocked visibility_854_private_controls_missing visibility_854 854
if grep -Fq 'context_state=baseline runtime_state=not-run-baseline' "$VISIBILITY_LOG"; then
  VISIBILITY_STATE=baseline
elif grep -Fq 'context_state=resolved runtime_state=pass' "$VISIBILITY_LOG"; then
  VISIBILITY_STATE=resolved
else
  cat "$VISIBILITY_LOG" >&2 || true
  blocked visibility_854_partial_or_unknown_state visibility_854 854
fi

printf 'MODULE_GRAPH_FACADE_VERTICAL_RECEIPT status=pass raw_authority=explicit_sha256 raw_sha256=%s vertical_behavior=pass graph_identity=unproven full_ir_input_receipt=closure_order_matched lowering=executed lowering_identity_consumption=unproven driver=canonical_full_ir legacy_compact=disabled phase_topology=closure_checker_count_full_ir_input_receipt_matched runtime_stdout=42_LF fixture_main_sha256=%s facade=selective negatives=3/3 missing_package=fail_closed missing_package_lowering_markers=absent fallback=none visibility_854=%s issue_991_oracle=%s issue_991_head=%s\n' \
  "$RAW_SHA256" "$VERTICAL_MAIN_SHA256" "$VISIBILITY_STATE" "$ISSUE_991_STATE" "$ISSUE_991_HEAD"
printf 'MODULE_GRAPH_FACADE_VERTICAL_PASS\n'
