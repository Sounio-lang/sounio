#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE="${PLACE_CANONICAL_BINDING_BASE_SHA:-90a8a83ac15bac6fc91e631d6847131d1b5fd257}"
SOUC="${SOUC_BIN:-}"
EXPECTED_COMPILER_SHA256="${PLACE_CANONICAL_BINDING_EXPECTED_COMPILER_SHA256:-}"
ORIGINAL="tests/native-v2/place_ir_legacy_nominal_collision_witness.sio"
SWAPPED="tests/native-v2/place_ir_legacy_nominal_collision_swapped_witness.sio"
IMPORT_AB="tests/native-v2/place_canonical_binding_shadow_modules/import_ab.sio"
IMPORT_BA="tests/native-v2/place_canonical_binding_shadow_modules/import_ba.sio"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-place-canonical-binding.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

fail() {
  printf 'PLACE_CANONICAL_BINDING_FAIL reason=%s\n' "$1" >&2
  exit 1
}

for file in \
  self-hosted/parser/ast.sio \
  self-hosted/check/types.sio \
  self-hosted/check/defs.sio \
  self-hosted/check/check.sio \
  self-hosted/ir/ir.sio \
  self-hosted/compiler/module_frontend.sio \
  "$ORIGINAL" \
  "$SWAPPED" \
  "$IMPORT_AB" \
  "$IMPORT_BA" \
  tests/native-v2/place_canonical_binding_shadow_modules/a/mod.sio \
  tests/native-v2/place_canonical_binding_shadow_modules/b/mod.sio; do
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

extract_struct() {
  local file="$1"
  local start="$2"
  local stop="$3"
  awk -v start="$start" -v stop="$stop" '
    index($0, start) { emit = 1 }
    emit { print }
    emit && index($0, stop) { exit }
  ' "$file"
}

# The multi-module frontend preserves an exact function route within one active
# load as (module index, name). The load-order-local index is not a durable DefId.
grep -Fq 'a.defining_module_id == b.defining_module_id' self-hosted/compiler/module_frontend.sio || fail function_route_identity_missing
grep -Fq 'if defining_module_id < 0 && defining_module_id != IR_DEFINING_MODULE_BUILTIN' self-hosted/compiler/module_frontend.sio || fail invalid_function_identity_not_rejected
grep -Fq 'candidate.defining_module_id == defining_module_id && ir_name_eq(candidate.name, name)' self-hosted/compiler/module_frontend.sio || fail exact_active_load_function_route_lookup_missing
grep -Fq 'if candidate.defining_module_id == IR_DEFINING_MODULE_AMBIGUOUS' self-hosted/compiler/module_frontend.sio || fail ambiguous_function_route_rejection_missing

# The first exact loss boundary: TypeExpr has a path, but named-type lowering
# copies only its head segment into a module-less TyNamed TypeEntry.
grep -Fq 'path: AstPath' self-hosted/parser/ast.sio || fail type_expr_path_missing
grep -Fq 'fn checker_copy_string_list_to_name(seg: StringList) -> Name' self-hosted/check/check.sio || fail named_path_collapse_helper_missing
grep -Fq 'let name = checker_copy_string_list_to_name(path.segments)' self-hosted/check/check.sio || fail named_type_path_not_collapsed
grep -Fq 'ty_named(name)' self-hosted/check/check.sio || fail named_type_lowering_missing

extract_struct self-hosted/check/types.sio 'pub struct TypeEntry {' 'pub struct TypeEntryList {' >"$TMP/type-entry.txt"
if grep -Eq 'defining_module_id|type_id|layout_id|generation|epoch' "$TMP/type-entry.txt"; then
  fail type_entry_claimed_canonical_identity
fi

# The checker then chooses the first same-name struct, while FieldInfo has no
# stable field identity. This gate must fail if that boundary moves.
grep -Fq 'let struct_idx = struct_table_find((*c).structs, base_ty.name)' self-hosted/check/check.sio || fail checker_field_name_lookup_boundary_moved
grep -Fq 'if name_eq((*t).entries[i].name, name)' self-hosted/check/defs.sio || fail struct_table_first_match_boundary_moved
grep -Fq 'let direct_sig = fn_sig_table_get((*c).fn_sigs, direct_sig_id)' self-hosted/check/check.sio || fail checker_call_resolution_receipt_source_missing
grep -Fq 'return effective_return_type' self-hosted/check/check.sio || fail checker_call_resolution_return_boundary_moved
grep -Fq 'let field_ty = field_info_list_find(si.fields, e.name)' self-hosted/check/check.sio || fail checker_field_resolution_receipt_source_missing
extract_struct self-hosted/check/defs.sio 'pub struct FieldInfo {' 'pub struct FieldInfoList {' >"$TMP/field-info.txt"
if grep -Eq 'field_id|ordinal|defining_module_id|layout_id|generation|epoch' "$TMP/field-info.txt"; then
  fail field_info_claimed_canonical_identity
fi

# Production lowering layout is likewise module-less. A caller cannot derive a
# canonical Place root/layout from this table without introducing a fallback.
extract_struct self-hosted/ir/ir.sio 'pub struct StructLayoutEntry {' 'pub struct StructLayoutTable {' >"$TMP/struct-layout.txt"
if grep -Eq 'defining_module_id|layout_id|generation|epoch' "$TMP/struct-layout.txt"; then
  fail struct_layout_claimed_canonical_identity
fi

for source in "$ORIGINAL" "$SWAPPED"; do
  grep -Fq 'make_b().shared' "$source" || fail "direct_call_projection_missing_${source//\//_}"
  grep -Fq 'typed.shared != 42' "$source" || fail "typed_control_missing_${source//\//_}"
  if grep -Eq 'place_alloc_root|place_append_field|field_ordinal|layout_id|type_id' "$source"; then
    fail "witness_supplies_place_authority_${source//\//_}"
  fi
done

for source in "$IMPORT_AB" "$IMPORT_BA"; do
  grep -Fq 'make_b().shared' "$source" || fail "imported_direct_call_projection_missing_${source//\//_}"
  grep -Fq 'typed.shared != 42' "$source" || fail "imported_typed_control_missing_${source//\//_}"
  if grep -Eq 'place_alloc_root|place_append_field|field_ordinal|layout_id|type_id' "$source"; then
    fail "imported_witness_supplies_place_authority_${source//\//_}"
  fi
done
grep -Fq 'pub struct A {' tests/native-v2/place_canonical_binding_shadow_modules/a/mod.sio || fail imported_struct_a_missing
grep -Fq 'pub struct B {' tests/native-v2/place_canonical_binding_shadow_modules/b/mod.sio || fail imported_struct_b_missing

{
  git diff --name-only "$BASE"
  git ls-files --others --exclude-standard
} | sed '/^$/d' | sort -u >"$TMP/actual-changed-files.txt"
printf '%s\n' \
  scripts/ci/place_canonical_binding_shadow_gate.sh \
  tests/native-v2/place_canonical_binding_shadow_modules/a/mod.sio \
  tests/native-v2/place_canonical_binding_shadow_modules/b/mod.sio \
  tests/native-v2/place_canonical_binding_shadow_modules/import_ab.sio \
  tests/native-v2/place_canonical_binding_shadow_modules/import_ba.sio \
  tests/native-v2/place_ir_legacy_nominal_collision_swapped_witness.sio \
  >"$TMP/allowed-changed-files.txt"
if ! diff -u "$TMP/allowed-changed-files.txt" "$TMP/actual-changed-files.txt" >"$TMP/changed-files.diff"; then
  cat "$TMP/changed-files.diff" >&2
  fail changed_file_allowlist_mismatch
fi
for default_surface in \
  self-hosted/ir/mod.sio \
  self-hosted/compiler/main.sio \
  scripts/bootstrap/bootstrap_concat.sh \
  bin/souc; do
  grep -Fq 'place_canonical_binding_shadow' "$default_surface" && fail "gate_imported_by_${default_surface//\//_}"
done

run_witness() {
  local label="$1"
  local source="$2"
  local expected_rc="$3"
  "$SOUC" compile "$source" -o "$TMP/$label.elf" >"$TMP/$label.build.log" 2>&1 || {
    cat "$TMP/$label.build.log" >&2
    fail "${label}_compile"
  }
  chmod +x "$TMP/$label.elf"
  set +e
  "$TMP/$label.elf" >"$TMP/$label.out" 2>&1
  local actual_rc=$?
  set -e
  [[ "$actual_rc" -eq "$expected_rc" ]] || {
    cat "$TMP/$label.out" >&2
    fail "${label}_expected_${expected_rc}_actual_${actual_rc}"
  }
  printf '%s' "$actual_rc"
}

original_rc="$(run_witness declaration_a_then_b "$ORIGINAL" 11)"
swapped_rc="$(run_witness declaration_b_then_a "$SWAPPED" 42)"
import_ab_rc="$(run_witness import_a_then_b "$IMPORT_AB" 11)"
import_ba_rc="$(run_witness import_b_then_a "$IMPORT_BA" 42)"
branch="$(git branch --show-current)"

printf '%s\n' 'PLACE_CANONICAL_BINDING_TRACE active_load_function_route=module_index_plus_name nominal_return_type=name_only field_identity=absent production_layout_identity=absent'
printf '%s\n' 'PLACE_CANONICAL_BINDING_NOMINAL_LOSS boundary=checker_lower_named_type_mut detail=TypeExpr.path_to_moduleless_TypeEntry.TyNamed'
printf '%s\n' 'PLACE_CANONICAL_BINDING_RECEIPT_LOSS boundary=checker_expression_return detail=callee_active_load_route,owner_struct,matched_field,ordinal_not_persisted first_wrong_choice=lower.field_idx_for_base_ref_non_ident default_pipeline_unchanged=true'
printf 'PLACE_CANONICAL_BINDING_DIFFERENTIAL single_module_typed_local=42 imported_typed_local_ab=42 imported_typed_local_ba=42 direct_call_a_then_b=%s direct_call_b_then_a=%s import_ab=%s import_ba=%s declaration_order_changes_result=true import_order_changes_result=true witness_place_authority_supplied=false legacy_fallback_observed=true\n' "$original_rc" "$swapped_rc" "$import_ab_rc" "$import_ba_rc"
printf '%s\n' 'PLACE_CANONICAL_BINDING_ADVERSARY same_field_spelling=observed declaration_order=fail import_order=fail nested_projection=not_exercised direct_call_projection=blocked unresolved=not_reached ambiguous=not_reached stale_layout_epoch=not_representable'
printf '%s\n' 'PLACE_CANONICAL_BINDING_BLOCKER Blocker-ID=BLK-20260715-place-canonical-nominal-identity Status=classified Severity=B1 Class=compiler-semantics Evidence-Level=E3 Result=BLOCKED_REQUIRED_METADATA_NOT_EXPORTED_ON_ACTIVE_FIELD_PATH Binder-Present=false Binder-Fallback-Used=not_applicable Missing=root_binding_receipt,module_qualified_type_identity,field_identity,target_layout_epoch'
printf 'PLACE_CANONICAL_BINDING_BLOCKER_SCOPE Owner=Codex-place-canonical-binding-shadow Lane=compiler-derived-canonical-Place-binding Worktree=%s Branch=%s Files-Owned=scripts/ci/place_canonical_binding_shadow_gate.sh,tests/native-v2/place_canonical_binding_shadow_modules,tests/native-v2/place_ir_legacy_nominal_collision_swapped_witness.sio Files-Read-Only=self-hosted/parser,self-hosted/check,self-hosted/ir,self-hosted/compiler Do-Not-Touch=parser,checker,lower,IR,SOIR,writer,backend\n' "$ROOT" "$branch"
printf '%s\n' 'PLACE_CANONICAL_BINDING_BLOCKER_CONTRACT Repro=SOUC_BIN=<source-fresh-elf>_PLACE_CANONICAL_BINDING_EXPECTED_COMPILER_SHA256=<sha256>_bash_scripts/ci/place_canonical_binding_shadow_gate.sh Observed=order-dependent_11_vs_42 Expected=derived_Place_result_42_independent_of_declaration_and_import_order Acceptance-Gate=future_snapshot_receipt_gate_then_this_differential Evidence=gate_stdout_or_CI_log Fallback-Path=none Legacy-Kept=yes LLM-Offload=not-required Next-Action=export_generation-bound_field-resolution-receipt_before_shared-multimodule-checker-free'
printf 'PLACE_CANONICAL_BINDING_PASS mode=executable_blocker compiler=%s compiler_sha256=%s base=%s\n' "$SOUC" "$compiler_sha256" "$BASE"
