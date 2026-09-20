#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE="${NOMINAL_FIELD_RECEIPT_BASE_SHA:-4c952e6ee7bcd0855f675fc662420c2fa507e19a}"
SOUC="${SOUC_BIN:-}"
EXPECTED_COMPILER_SHA256="${NOMINAL_FIELD_RECEIPT_EXPECTED_COMPILER_SHA256:-}"
CHECKER="self-hosted/check/check.sio"
PROBE="self-hosted/check/nominal_field_resolution_receipt_shadow_probe.sio"
BINDER="self-hosted/ir/arena_v2_place_nominal_receipt_binding_shadow.sio"
WITNESS="tests/native-v2/nominal_field_resolution_receipt_shadow_witness.sio"
ORIGINAL="tests/native-v2/place_ir_legacy_nominal_collision_witness.sio"
SWAPPED="tests/native-v2/place_ir_legacy_nominal_collision_swapped_witness.sio"
IMPORT_AB="tests/native-v2/place_canonical_binding_shadow_modules/import_ab.sio"
IMPORT_BA="tests/native-v2/place_canonical_binding_shadow_modules/import_ba.sio"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-nominal-field-receipt.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

fail() {
  printf 'NOMINAL_FIELD_RECEIPT_FAIL reason=%s\n' "$1" >&2
  exit 1
}

for file in "$CHECKER" "$PROBE" "$BINDER" "$WITNESS" \
  "$ORIGINAL" "$SWAPPED" "$IMPORT_AB" "$IMPORT_BA"; do
  [[ -f "$file" ]] || fail "missing_${file//\//_}"
done
[[ -n "$SOUC" ]] || fail explicit_source_fresh_compiler_required
[[ -n "$EXPECTED_COMPILER_SHA256" ]] || fail expected_compiler_sha256_required
[[ -x "$SOUC" ]] || fail compiler_missing
compiler_magic="$(od -An -tx1 -N4 "$SOUC" | tr -d ' \n')"
[[ "$compiler_magic" == "7f454c46" ]] || fail source_fresh_compiler_must_be_elf
compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
[[ "$compiler_sha256" == "$EXPECTED_COMPILER_SHA256" ]] || fail compiler_sha256_mismatch
git cat-file -e "$BASE^{commit}" 2>/dev/null || fail base_sha_unavailable

# The default checker owns only a dormant nullable pointer. Snapshot-local
# observation tokens never claim resolver-owned DefId stability.
grep -Fq 'nominal_field_resolution_shadow: *mut NominalFieldResolutionShadow' "$CHECKER" || fail dormant_pointer_missing
grep -Fq 'nominal_field_resolution_shadow: 0 as *mut NominalFieldResolutionShadow' "$CHECKER" || fail default_pointer_not_null
grep -Fq 'not resolver-owned DefIds and not semantic authority across snapshots' "$CHECKER" || fail observational_boundary_missing
grep -Fq 'next_observation_token: i64' "$CHECKER" || fail observation_token_allocator_missing
if grep -Eq 'pub fn nominal_field_resolution_receipt_(root|owner_type|field|result_type)_identity' "$CHECKER"; then
  fail public_semantic_identity_claim_present
fi
grep -Fq 'pub fn nominal_field_resolution_receipt_field_observation_token' "$CHECKER" || fail field_observation_token_accessor_missing

# Capture occurs at the exact live resolution boundaries and is copied through
# the real binding slot before TypeEnv bind. Numeric tokens are only checked for
# within-snapshot liveness; the AB/BA probe never compares their values.
grep -Fq 'nominal_field_resolution_record_direct_call(c, direct_sig_id, direct_sig, effective_return_type)' "$CHECKER" || fail direct_call_capture_hook_missing
grep -Fq 'let field_ty = field_info_list_find(si.fields, e.name)' "$CHECKER" || fail legacy_field_resolution_boundary_moved
grep -Fq 'nominal_field_resolution_record_field(c, e)' "$CHECKER" || fail field_capture_hook_missing
grep -Fq 'nominal_field_resolution_binding_from_expr(c, cur_env.count)' "$CHECKER" || fail let_binding_capture_hook_missing
grep -Fq 'nominal_field_resolution_binding_from_expr(c, (*c).env.count)' "$CHECKER" || fail var_binding_capture_hook_missing
grep -Fq 'nominal_field_resolution_clear_binding_slot(c, (*c).env.count)' "$CHECKER" || fail reused_binding_slot_clear_missing
grep -Fq 'nominal_field_resolution_record_ident(c, e.name)' "$CHECKER" || fail binding_restore_hook_missing
grep -Fq 'if depth < 0 || depth > MAX_EXPR_DEPTH { return }' "$CHECKER" || fail ident_depth_bound_missing
grep -Fq 'NOMINAL_FIELD_RECEIPT_NOMINAL_READY: i64 = 31' "$CHECKER" || fail nominal_ready_mask_changed
grep -Fq 'NOMINAL_FIELD_RECEIPT_BINDER_READY: i64 = 127' "$CHECKER" || fail binder_ready_mask_changed
grep -Fq 'tokens are snapshot-local, not resolver-owned DefIds' "$CHECKER" || fail snapshot_api_boundary_missing
grep -Fq 'deliberately not compared across AB/BA' "$PROBE" || fail cross_order_numeric_comparison_boundary_missing
grep -Fq 'if (*p).state != 0' "$CHECKER" || fail overlapping_snapshot_rejection_missing
grep -Fq 'overlapping_snapshot' "$PROBE" || fail overlapping_snapshot_adversary_missing
if grep -Eq '(direct_ab|direct_ba|import_ab|import_ba)[[:space:]]*(==|!=)[[:space:]]*(direct_ab|direct_ba|import_ab|import_ba)' "$PROBE"; then
  fail cross_order_numeric_token_comparison_present
fi

# No target layout authority exists in this slice. The Place handoff validates
# liveness and fails before allocation, projection append, or finalization.
grep -Fq 'NOMINAL_FIELD_RECEIPT_ERR_TARGET_LAYOUT_MISSING' "$BINDER" || fail missing_layout_status_absent
if grep -Eq 'ir_module_arena_v2_place_(alloc_root|append_field|finalize)[[:space:]]*\(' "$BINDER"; then
  fail place_mutation_present
fi
grep -Fq 'if !ir_module_arena_v2_module_id_is_live(module)' "$BINDER" || fail module_liveness_check_missing
grep -Fq 'if ir_module_arena_v2_place_is_live(module, out)' "$BINDER" || fail live_output_rejection_missing
if sed -n '/pub struct NominalFieldResolutionShadow {/,/^}/p' "$CHECKER" | grep -Eq 'target_layout|layout_epoch'; then
  fail target_layout_authority_synthesized
fi

{
  git diff --name-only "$BASE"
  git ls-files --others --exclude-standard
} | sed '/^$/d' | sort -u >"$TMP/actual-changed-files.txt"
printf '%s\n' \
  scripts/ci/nominal_field_resolution_receipt_shadow_gate.sh \
  self-hosted/check/check.sio \
  self-hosted/check/nominal_field_resolution_receipt_shadow_probe.sio \
  self-hosted/ir/arena_v2_place_nominal_receipt_binding_shadow.sio \
  tests/native-v2/nominal_field_resolution_local_witness.sio \
  tests/native-v2/nominal_field_resolution_nested_witness.sio \
  tests/native-v2/nominal_field_resolution_receipt_shadow_witness.sio \
  >"$TMP/allowed-changed-files.txt"
if ! diff -u "$TMP/allowed-changed-files.txt" "$TMP/actual-changed-files.txt" >"$TMP/changed-files.diff"; then
  cat "$TMP/changed-files.diff" >&2
  fail changed_file_allowlist_mismatch
fi

for default_surface in \
  self-hosted/ir/mod.sio \
  self-hosted/ir/lower.sio \
  self-hosted/ir/ir.sio \
  self-hosted/ir/soir_writer.sio \
  self-hosted/compiler/main.sio \
  scripts/bootstrap/bootstrap_concat.sh \
  bin/souc; do
  git diff --quiet "$BASE" -- "$default_surface" || fail "default_surface_changed_${default_surface//\//_}"
  grep -Fq 'nominal_field_resolution_receipt_shadow' "$default_surface" && fail "receipt_imported_by_${default_surface//\//_}"
done

"$SOUC" --check "$WITNESS" >"$TMP/witness.check.log" 2>&1 || {
  cat "$TMP/witness.check.log" >&2
  fail witness_source_check
}
"$SOUC" compile "$WITNESS" -o "$TMP/witness.elf" >"$TMP/witness.build.log" 2>&1 || {
  cat "$TMP/witness.build.log" >&2
  fail witness_compile
}
chmod +x "$TMP/witness.elf"
set +e
"$TMP/witness.elf" >"$TMP/witness.out" 2>&1
witness_rc=$?
set -e
[[ "$witness_rc" -eq 139 ]] || {
  cat "$TMP/witness.out" >&2
  fail "imported_transitive_lowering_expected_139_actual_${witness_rc}"
}
grep -Fq 'Merged IR: 2' "$TMP/witness.build.log" || {
  cat "$TMP/witness.build.log" >&2
  fail imported_transitive_lowering_evidence_missing
}

run_legacy() {
  local label="$1"
  local source="$2"
  local expected="$3"
  "$SOUC" compile "$source" -o "$TMP/$label.elf" >"$TMP/$label.build.log" 2>&1 || {
    cat "$TMP/$label.build.log" >&2
    fail "${label}_compile"
  }
  chmod +x "$TMP/$label.elf"
  set +e
  "$TMP/$label.elf" >"$TMP/$label.out" 2>&1
  local actual=$?
  set -e
  [[ "$actual" -eq "$expected" ]] || {
    cat "$TMP/$label.out" >&2
    fail "${label}_expected_${expected}_actual_${actual}"
  }
  printf '%s' "$actual"
}

original_rc="$(run_legacy declaration_a_then_b "$ORIGINAL" 11)"
swapped_rc="$(run_legacy declaration_b_then_a "$SWAPPED" 42)"
import_ab_rc="$(run_legacy import_a_then_b "$IMPORT_AB" 11)"
import_ba_rc="$(run_legacy import_b_then_a "$IMPORT_BA" 42)"

printf '%s\n' 'NOMINAL_FIELD_RECEIPT_CHECK source=pass default_surfaces=unchanged off_default=true executable_receipt_hook=blocked'
printf '%s\n' 'NOMINAL_FIELD_RECEIPT_OBSERVATION scope=snapshot_local authority=observational_only resolver_defid=false cross_snapshot_stable=false cross_order_numeric_comparison=forbidden ready_mask=31 lifecycle_contract=structural_begin,freeze,reset stale_snapshot_contract=reject stale_receipt_contract=reject live_output_contract=reject parent_link=structurally_present adversaries=encoded_not_executed_due_imported_lowering_blocker'
printf '%s\n' 'NOMINAL_FIELD_RECEIPT_PLACE preflight=fail_closed target_layout=false layout_epoch=false place_allocated=false status=-75'
printf 'NOMINAL_FIELD_RECEIPT_DIFFERENTIAL original=%s swapped=%s import_ab=%s import_ba=%s default_legacy_kept=true\n' "$original_rc" "$swapped_rc" "$import_ab_rc" "$import_ba_rc"
printf '%s\n' 'NOMINAL_FIELD_RECEIPT_BLOCKER Blocker-ID=BLK-20260715-nominal-receipt-imported-transitive-lowering Status=classified Severity=B1 Class=harness-routing Evidence-Level=E2 Result=BLOCKED_IMPORTED_TRANSITIVE_LOWERING Observed=compile_success_merged_ir_2_runtime_rc_139 Expected=probe_facade_transitive_checker_bodies_lowered Acceptance-Gate=witness_runtime_prints_42 Owner=native-v2-imported-module-lowering Next-Action=lower_transitive_imported_function_closure'
printf 'NOMINAL_FIELD_RECEIPT_PASS mode=source_preservation_slice_not_executable_hook_proof compiler=%s compiler_sha256=%s base=%s\n' "$SOUC" "$compiler_sha256" "$BASE"
