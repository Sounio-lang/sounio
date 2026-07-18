#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PINNED_BASE="4a63c2ba16b4aecf85759bf3f3a7ba9105c79986"
REQUESTED_BASE="${IR_MODULE_ARENA_V2_SOIR_BASE_SHA:-$PINNED_BASE}"
BASE="$PINNED_BASE"
SOUC="${SOUC_BIN:-$ROOT/bin/madaros}"
ARENA="self-hosted/ir/arena_v2_shadow.sio"
WRITER="self-hosted/ir/soir_writer.sio"
BRIDGE="self-hosted/ir/arena_v2_soir_bridge.sio"
PROVENANCE_MODE="${IR_MODULE_ARENA_V2_SOIR_PROVENANCE_MODE:-git}"
GIT_SCOPE="${IR_MODULE_ARENA_V2_SOIR_GIT_SCOPE:-current_tree}"
CONTENT_MANIFEST="${IR_MODULE_ARENA_V2_SOIR_CONTENT_MANIFEST:-}"
EXPECTED_MANIFEST_SHA256="${IR_MODULE_ARENA_V2_SOIR_EXPECTED_MANIFEST_SHA256:-}"
RUNTIME_TIMEOUT_SECONDS="${IR_MODULE_ARENA_V2_SOIR_RUNTIME_TIMEOUT_SECONDS:-15}"
TIMEOUT_SELFTEST="${IR_MODULE_ARENA_V2_SOIR_TIMEOUT_SELFTEST:-1}"
PROVENANCE_RECEIPT=""
CONTENT_MANIFEST_SHA256=""
HEAD_SHA="not_available"
TREE_SHA="not_available"
BASE_RECEIPT="not_checked_manifest_scope"
PRECISION_RECEIPT="not_checked_manifest_scope"
LEGACY_RECEIPT="not_checked_manifest_scope"
LANE_CONTENT_SHA256=""
GIT_SCOPE_RECEIPT="not_applicable_manifest"
EXACT_WRITE_SET_RECEIPT="not_checked_manifest_scope"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-ir-module-arena-v2-soir-v5.XXXXXX")"

cleanup() {
  rm -rf "$TMP"
}
trap cleanup EXIT

fail() {
  printf 'IR_MODULE_ARENA_V2_SOIR_V5_FAIL reason=%s\n' "$1" >&2
  exit 1
}

progress() {
  printf 'IR_MODULE_ARENA_V2_SOIR_V5_PROGRESS stage=%s subject=%s status=%s\n' \
    "$1" "$2" "$3" >&2
}

run_runtime_with_timeout() {
  local elf="$1"
  local out="$2"
  local timeout_seconds="$3"
  timeout --signal=TERM --kill-after=2s "${timeout_seconds}s" "$elf" >"$out" 2>&1
}

run_witness_runtime() {
  local name="$1"
  local elf="$2"
  local out="$3"
  local timeout_seconds="$4"
  local runtime_rc

  progress composite_runtime "$name" begin
  set +e
  run_runtime_with_timeout "$elf" "$out" "$timeout_seconds"
  runtime_rc=$?
  set -e
  if [[ "$runtime_rc" -eq 124 ]]; then
    progress composite_runtime "$name" timeout
    cat "$out" >&2
    fail "composite_runtime_timeout_$name"
  fi
  if [[ "$runtime_rc" -ne 0 ]]; then
    progress composite_runtime "$name" fail
    cat "$out" >&2
    fail "composite_runtime_${name}_rc_${runtime_rc}"
  fi
  progress composite_runtime "$name" pass
}

run_timeout_selftest() {
  local source="$TMP/runtime_timeout_selftest.sio"
  local elf="$TMP/runtime_timeout_selftest.elf"
  local build_log="$TMP/runtime_timeout_selftest.build.log"
  local run_log="$TMP/runtime_timeout_selftest.out"
  local classification_log="$TMP/runtime_timeout_selftest.classification.log"
  local classification_rc

  printf '%s\n' \
    'module runtime_timeout_selftest' \
    '' \
    'fn main() -> i64 {' \
    '    while true {}' \
    '    0' \
    '}' >"$source"

  progress timeout_selftest forced_hang build_begin
  "$SOUC" --native-v2-compile "$source" -o "$elf" >"$build_log" 2>&1 || {
    cat "$build_log" >&2
    fail timeout_selftest_build
  }
  chmod +x "$elf"
  progress timeout_selftest forced_hang runtime_begin
  set +e
  (run_witness_runtime timeout_selftest "$elf" "$run_log" 1) \
    >"$classification_log" 2>&1
  classification_rc=$?
  set -e
  if [[ "$classification_rc" -ne 1 ]] ||
     ! grep -Fxq 'IR_MODULE_ARENA_V2_SOIR_V5_FAIL reason=composite_runtime_timeout_timeout_selftest' "$classification_log"; then
    cat "$classification_log" >&2
    fail "timeout_selftest_classification_rc_${classification_rc}"
  fi
  progress timeout_selftest forced_hang timeout_classified
  printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_TIMEOUT_SELFTEST forced_hang=timeout runtime_rc=124 fail_closed=pass seconds=1'
}

expected_write_set() {
  printf '%s\n' \
    scripts/ci/ir_module_arena_v2_soir_v5_bridge_gate.sh \
    self-hosted/ir/arena_v2_shadow.sio \
    self-hosted/ir/arena_v2_soir_bridge.sio \
    self-hosted/ir/soir_writer.sio \
    tests/native-v2/ir_module_arena_v2_soir_v5_bridge_cross_arena_witness.sio \
    tests/native-v2/ir_module_arena_v2_soir_v5_bridge_invalid_witness.sio \
    tests/native-v2/ir_module_arena_v2_soir_v5_bridge_mutation_preflight_probe.sio \
    tests/native-v2/ir_module_arena_v2_soir_v5_bridge_mutation_witness.sio \
    tests/native-v2/ir_module_arena_v2_soir_v5_bridge_plan_lifecycle_witness.sio \
    tests/native-v2/ir_module_arena_v2_soir_v5_bridge_reuse_witness.sio \
    tests/native-v2/ir_module_arena_v2_soir_v5_bridge_sequence_probe.sio \
    tests/native-v2/ir_module_arena_v2_soir_v5_bridge_stale_witness.sio \
    tests/native-v2/ir_module_arena_v2_soir_v5_bridge_witness.sio
}

validate_git_provenance() {
  local -a expected_paths
  local dirty_paths

  HEAD_SHA="$(git rev-parse HEAD)"
  TREE_SHA="$(git rev-parse 'HEAD^{tree}')"

  mapfile -t expected_paths < <(expected_write_set)
  for expected_path in "${expected_paths[@]}"; do
    git ls-files --error-unmatch -- "$expected_path" >/dev/null 2>&1 \
      || fail expected_write_set_path_untracked
  done
  dirty_paths="$(git status --porcelain --untracked-files=all -- "${expected_paths[@]}")"
  if [[ -n "$dirty_paths" ]]; then
    printf '%s\n' "$dirty_paths" >&2
    fail expected_write_set_dirty
  fi

  case "$GIT_SCOPE" in
    current_tree)
      [[ -z "${IR_MODULE_ARENA_V2_SOIR_BASE_SHA:-}" ]] || fail base_not_applicable_current_tree
      PROVENANCE_RECEIPT="git_head_tree_bound_current_tree"
      BASE_RECEIPT="not_checked_current_tree"
      GIT_SCOPE_RECEIPT="current_tree"
      EXACT_WRITE_SET_RECEIPT="not_checked_current_tree"
      ;;
    publication)
      [[ "$REQUESTED_BASE" == "$PINNED_BASE" ]] || fail base_override_rejected
      git cat-file -e "$BASE^{commit}" 2>/dev/null || fail base_sha_unavailable
      [[ "$HEAD_SHA" != "$BASE" ]] || fail head_equals_pinned_base
      git merge-base --is-ancestor "$BASE" "$HEAD_SHA" || fail pinned_base_not_ancestor

      git diff --name-only "$BASE" "$HEAD_SHA" | LC_ALL=C sort >"$TMP/write-set.actual"
      expected_write_set | LC_ALL=C sort >"$TMP/write-set.expected"
      diff -u "$TMP/write-set.expected" "$TMP/write-set.actual" \
        >"$TMP/write-set.diff" 2>&1 || {
          cat "$TMP/write-set.diff" >&2
          fail write_set_expanded
        }

      PROVENANCE_RECEIPT="git_head_tree_bound_publication"
      BASE_RECEIPT="$BASE"
      GIT_SCOPE_RECEIPT="publication_branch_exact"
      EXACT_WRITE_SET_RECEIPT="pass"
      ;;
    *)
      fail git_scope_invalid
      ;;
  esac
}

validate_manifest_provenance() {
  [[ -n "$CONTENT_MANIFEST" ]] || fail manifest_path_required
  [[ -n "$EXPECTED_MANIFEST_SHA256" ]] || fail manifest_sha256_required
  [[ "$EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail manifest_sha256_malformed
  [[ -f "$CONTENT_MANIFEST" ]] || fail manifest_missing
  [[ -s "$CONTENT_MANIFEST" ]] || fail manifest_empty

  CONTENT_MANIFEST_SHA256="$(sha256sum "$CONTENT_MANIFEST" | awk '{print $1}')"
  [[ "$CONTENT_MANIFEST_SHA256" == "$EXPECTED_MANIFEST_SHA256" ]] || fail manifest_sha256_mismatch

  set +e
  LC_ALL=C grep -Env '^[0-9a-f]{64}  [A-Za-z0-9._/-]+$' "$CONTENT_MANIFEST" \
    >"$TMP/manifest-format.log" 2>&1
  manifest_format_rc=$?
  set -e
  if [[ "$manifest_format_rc" -eq 0 ]]; then
    cat "$TMP/manifest-format.log" >&2
    fail manifest_format_invalid
  fi
  [[ "$manifest_format_rc" -eq 1 ]] || fail manifest_format_scan_error
  [[ "$(wc -l <"$CONTENT_MANIFEST" | tr -d ' ')" == 13 ]] || fail manifest_entry_count_not_13

  awk '{print $2}' "$CONTENT_MANIFEST" | LC_ALL=C sort >"$TMP/manifest-paths.actual"
  expected_write_set | LC_ALL=C sort >"$TMP/manifest-paths.expected"
  diff -u "$TMP/manifest-paths.expected" "$TMP/manifest-paths.actual" \
    >"$TMP/manifest-paths.diff" 2>&1 || {
      cat "$TMP/manifest-paths.diff" >&2
      fail manifest_write_set_mismatch
    }

  sha256sum -c "$CONTENT_MANIFEST" >"$TMP/manifest-content-check.log" 2>&1 || {
    cat "$TMP/manifest-content-check.log" >&2
    fail manifest_content_mismatch
  }
  PROVENANCE_RECEIPT="manifest_pinned_lane_content"
}

case "$PROVENANCE_MODE" in
  git)
    command -v git >/dev/null 2>&1 || fail git_required_for_git_provenance
    validate_git_provenance
    ;;
  manifest)
    validate_manifest_provenance
    ;;
  *)
    fail provenance_mode_invalid
    ;;
esac

for file in "$ARENA" "$WRITER" "$BRIDGE"; do
  [[ -f "$file" ]] || fail "missing_${file//\//_}"
done
[[ -x "$SOUC" ]] || fail compiler_missing
command -v timeout >/dev/null 2>&1 || fail timeout_command_missing
[[ "$RUNTIME_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail runtime_timeout_invalid
if (( RUNTIME_TIMEOUT_SECONDS > 60 )); then fail runtime_timeout_exceeds_cap_60; fi
[[ "$TIMEOUT_SELFTEST" == "1" ]] || fail timeout_selftest_required

grep -Fq 'pub let SOIR_WRITER_SCALAR_EMPTY_V5_SIZE: i64 = 320' "$WRITER" || fail writer_size_contract_missing
grep -Fq 'pub fn soir_writer_preflight_scalar_empty_module_v5(' "$WRITER" || fail writer_preflight_missing
grep -Fq 'pub fn soir_writer_emit_scalar_empty_module_v5(' "$WRITER" || fail writer_emit_missing
grep -Fq 'pub struct IrModuleArenaV2ModuleId {' "$ARENA" || fail module_id_missing
grep -Fq 'pub let IR_MODULE_ARENA_V2_MODULE_CAPACITY: i64 = 2' "$ARENA" || fail module_capacity_not_two
grep -Fq 'pub struct IrModuleArenaV2SoirPlanId {' "$BRIDGE" || fail plan_id_missing
grep -Fq 'pub let IR_MODULE_ARENA_V2_SOIR_PLAN_CAPACITY: i64 = 2' "$BRIDGE" || fail plan_capacity_not_two
grep -Fq 'pub fn ir_module_arena_v2_soir_preflight_empty_v5(' "$BRIDGE" || fail bridge_preflight_missing
grep -Fq 'pub fn ir_module_arena_v2_soir_emit_empty_v5(' "$BRIDGE" || fail bridge_emit_missing
grep -Fq 'soir_writer_preflight_scalar_empty_module_v5' "$BRIDGE" || fail writer_preflight_not_delegated
grep -Fq 'soir_writer_emit_scalar_empty_module_v5' "$BRIDGE" || fail writer_emit_not_delegated

dependency_pattern='use (parser|ir::ir)|\b(IrModule|IrInstr|TyF128|TyF256|numeric_payload)\b'
set +e
if command -v rg >/dev/null 2>&1; then
  rg -n "$dependency_pattern" "$ARENA" "$WRITER" "$BRIDGE" \
    >"$TMP/dependency-leak.log" 2>&1
else
  grep -En "$dependency_pattern" "$ARENA" "$WRITER" "$BRIDGE" \
    >"$TMP/dependency-leak.log" 2>&1
fi
dependency_scan_rc=$?
set -e
if [[ "$dependency_scan_rc" -eq 0 ]]; then
  cat "$TMP/dependency-leak.log" >&2
  fail shadow_dependency_expanded
fi
if [[ "$dependency_scan_rc" -ne 1 ]]; then
  cat "$TMP/dependency-leak.log" >&2
  fail dependency_scan_error
fi
if grep -Eq '(out_buf|buf):[[:space:]]*\[i8;[[:space:]]*131072\]' "$BRIDGE"; then
  fail buffer_passed_by_value
fi
if grep -Eq '^pub var IR_MODULE_ARENA_V2_SOIR_PLAN_' "$BRIDGE"; then
  fail plan_columns_public
fi

PROTECTED=(
  self-hosted/check/check.sio
  self-hosted/compiler/main.sio
  self-hosted/compiler/module_frontend.sio
  self-hosted/ir/lower.sio
  self-hosted/ir/ir.sio
  self-hosted/ir/serialize.sio
  self-hosted/ir/mod.sio
  self-hosted/ir/numeric_payload.sio
  self-hosted/ir/numeric_payload_wire.sio
  self-hosted/compiler/f128_f256_format_descriptor_probe.sio
  self-hosted/compiler/f128_f256_numeric_payload_probe.sio
  self-hosted/compiler/f128_f256_numeric_wire_probe.sio
  scripts/ci/madaros_f128_f256_format_identity_gate.sh
  scripts/ci/madaros_f128_f256_numeric_payload_gate.sh
  scripts/ci/madaros_f128_f256_numeric_wire_gate.sh
)
if [[ "$PROVENANCE_MODE" == "git" ]]; then
  for protected in "${PROTECTED[@]}"; do
    [[ -e "$protected" ]] || continue
    if [[ "$GIT_SCOPE" == "publication" ]]; then
      git diff --quiet "$BASE" -- "$protected" || fail "protected_surface_changed_${protected//\//_}"
    fi
    [[ -z "$(git status --short -- "$protected")" ]] || fail "protected_surface_dirty_${protected//\//_}"
  done

  if git status --porcelain --untracked-files=all | cut -c4- | grep -Eiq '(^|/)contest'; then
    fail contest_surface_dirty
  fi
fi

for default_surface in \
  self-hosted/ir/serialize.sio \
  self-hosted/ir/mod.sio \
  self-hosted/ir/lower.sio \
  self-hosted/check/check.sio \
  self-hosted/compiler/main.sio \
  self-hosted/compiler/module_frontend.sio; do
  [[ -f "$default_surface" ]] || continue
  if grep -Eq 'arena_v2_(shadow|soir_bridge)|ir::soir_writer' "$default_surface"; then
    fail "shadow_imported_by_${default_surface//\//_}"
  fi
done

if [[ "$PROVENANCE_MODE" == "git" ]]; then
  bash scripts/ci/madaros_f128_f256_format_identity_gate.sh --structural-only \
    >"$TMP/precision-identity.log" 2>&1 || {
      cat "$TMP/precision-identity.log" >&2
      fail precision_identity_regression
    }
  PRECISION_RECEIPT="preserved"
  LEGACY_RECEIPT="default"
fi

run_timeout_selftest

for source in "$WRITER" "$ARENA" "$BRIDGE"; do
  source_name="$(basename "$source" .sio)"
  progress source_check "$source_name" begin
  "$SOUC" check "$source" >"$TMP/$(basename "$source").check.log" 2>&1 || {
    cat "$TMP/$(basename "$source").check.log" >&2
    fail "source_check_${source//\//_}"
  }
  progress source_check "$source_name" pass
done

compose_and_run() {
  local witness="$1"
  local marker="$2"
  local name composite elf log
  name="$(basename "$witness" .sio)"
  composite="$TMP/$name.sio"
  elf="$TMP/$name.elf"
  log="$TMP/$name.log"

  progress composite "$name" begin

  {
    printf 'module ir::%s_gate\n\n' "$name"
    sed '/^module ir::soir_writer$/d' "$WRITER"
    sed '/^module ir::arena_v2_shadow$/d' "$ARENA"
    sed -e '/^module ir::arena_v2_soir_bridge$/d' \
        -e '/^use ir::arena_v2_shadow::\*$/d' \
        -e '/^use ir::soir_writer::\*$/d' "$BRIDGE"
    sed -e '/^use ir::arena_v2_shadow::\*$/d' \
        -e '/^use ir::soir_writer::\*$/d' \
        -e '/^use ir::arena_v2_soir_bridge::\*$/d' "$witness"
  } >"$composite"

  progress composite_check "$name" begin
  "$SOUC" check "$composite" >"$log" 2>&1 || {
    cat "$log" >&2
    fail "composite_check_$name"
  }
  progress composite_check "$name" pass
  progress composite_build "$name" begin
  "$SOUC" --native-v2-compile "$composite" -o "$elf" >>"$log" 2>&1 || {
    cat "$log" >&2
    fail "composite_build_$name"
  }
  chmod +x "$elf"
  progress composite_build "$name" pass
  run_witness_runtime "$name" "$elf" "$TMP/$name.out" "$RUNTIME_TIMEOUT_SECONDS"
  grep -Fxq "$marker" "$TMP/$name.out" || {
    cat "$TMP/$name.out" >&2
    fail "marker_missing_$name"
  }
  progress composite "$name" pass
}

compose_and_run tests/native-v2/ir_module_arena_v2_soir_v5_bridge_witness.sio IR_MODULE_ARENA_V2_SOIR_V5_CANONICAL_PASS
compose_and_run tests/native-v2/ir_module_arena_v2_soir_v5_bridge_invalid_witness.sio IR_MODULE_ARENA_V2_SOIR_V5_INVALID_PASS
compose_and_run tests/native-v2/ir_module_arena_v2_soir_v5_bridge_stale_witness.sio IR_MODULE_ARENA_V2_SOIR_V5_STALE_PASS
compose_and_run tests/native-v2/ir_module_arena_v2_soir_v5_bridge_reuse_witness.sio IR_MODULE_ARENA_V2_SOIR_V5_REUSE_PASS
compose_and_run tests/native-v2/ir_module_arena_v2_soir_v5_bridge_cross_arena_witness.sio IR_MODULE_ARENA_V2_SOIR_V5_CROSS_ARENA_PASS
compose_and_run tests/native-v2/ir_module_arena_v2_soir_v5_bridge_mutation_witness.sio IR_MODULE_ARENA_V2_SOIR_V5_MUTATION_PASS
compose_and_run tests/native-v2/ir_module_arena_v2_soir_v5_bridge_sequence_probe.sio IR_MODULE_ARENA_V2_SOIR_V5_SEQUENCE_PASS
compose_and_run tests/native-v2/ir_module_arena_v2_soir_v5_bridge_mutation_preflight_probe.sio IR_MODULE_ARENA_V2_SOIR_V5_MUTATION_PREFLIGHT_PASS
compose_and_run tests/native-v2/ir_module_arena_v2_soir_v5_bridge_plan_lifecycle_witness.sio IR_MODULE_ARENA_V2_SOIR_V5_PLAN_LIFECYCLE_PASS

grep -Fq 'SEQUENCE_AFTER_STALE status=-20' "$TMP/ir_module_arena_v2_soir_v5_bridge_sequence_probe.out" || fail stale_sequence_receipt_missing
grep -Fq 'SEQUENCE_AFTER_REUSE status=-22' "$TMP/ir_module_arena_v2_soir_v5_bridge_sequence_probe.out" || fail reuse_sequence_receipt_missing
grep -Fq 'canary=1' "$TMP/ir_module_arena_v2_soir_v5_bridge_sequence_probe.out" || fail sequence_canary_receipt_missing
grep -Fq 'MUTATION_PREFLIGHT status=-21 bss=8 id_live=1' "$TMP/ir_module_arena_v2_soir_v5_bridge_mutation_preflight_probe.out" || fail mutation_preflight_receipt_missing
grep -Fq 'MODULE_OUTPUT_LIVE status=-8 live_count=1 id_live=1' "$TMP/ir_module_arena_v2_soir_v5_bridge_reuse_witness.out" || fail module_output_live_receipt_missing

while IFS= read -r lane_path; do
  sha256sum "$lane_path"
done < <(expected_write_set) >"$TMP/lane-content.sha256"
LANE_CONTENT_SHA256="$(sha256sum "$TMP/lane-content.sha256" | awk '{print $1}')"

compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
printf 'IR_MODULE_ARENA_V2_SOIR_V5_WATCHDOG runtime_timeout_seconds=%s kill_after_seconds=2 timeout_rc=124 selftest=pass fail_closed=pass\n' \
  "$RUNTIME_TIMEOUT_SECONDS"
printf 'IR_MODULE_ARENA_V2_SOIR_V5_CHECK source=pass composite_matrix=9/9 precision_identity=%s provenance=%s\n' \
  "$PRECISION_RECEIPT" "$PROVENANCE_RECEIPT"
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_PLAN storage=module_owned_scalar_columns capacity=2 identity=slot,generation field_privacy=enforcement_not_claimed integrity=emit_revalidated binding=arena,module_slot,module_generation,mutation_epoch,start,capacity,required,end,version'
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_EMIT deterministic=pass wire=v5_empty_320 capacity_below=reject capacity_exact=pass no_partial_write=pass origin=zero_only'
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_IDS invalid=reject stale=reject reuse=reject module_output_live=reject cross_arena=reject mutation_after_preflight=reject mutation_repreflight=pass'
printf 'IR_MODULE_ARENA_V2_SOIR_V5_DIFFERENTIAL mode=shadow_canonical_not_differential byte_parity=not_claimed legacy=%s\n' "$LEGACY_RECEIPT"
if [[ "$PROVENANCE_MODE" == "manifest" ]]; then
  printf 'IR_MODULE_ARENA_V2_SOIR_V5_PROVENANCE provenance=manifest_pinned_lane_content manifest_sha256=%s entries=13 scope=lane_content_only protected_surfaces=not_checked\n' "$CONTENT_MANIFEST_SHA256"
else
  printf 'IR_MODULE_ARENA_V2_SOIR_V5_PROVENANCE provenance=%s scope=%s base=%s head=%s tree=%s exact_write_set=%s expected_paths_clean=pass\n' \
    "$PROVENANCE_RECEIPT" "$GIT_SCOPE_RECEIPT" "$BASE_RECEIPT" "$HEAD_SHA" "$TREE_SHA" "$EXACT_WRITE_SET_RECEIPT"
fi
printf 'IR_MODULE_ARENA_V2_SOIR_V5_PASS compiler=%s compiler_sha256=%s base=%s head=%s tree=%s lane_content_sha256=%s provenance=%s scope=%s exact_write_set=%s\n' \
  "$SOUC" "$compiler_sha256" "$BASE_RECEIPT" "$HEAD_SHA" "$TREE_SHA" "$LANE_CONTENT_SHA256" "$PROVENANCE_RECEIPT" "$GIT_SCOPE_RECEIPT" "$EXACT_WRITE_SET_RECEIPT"
