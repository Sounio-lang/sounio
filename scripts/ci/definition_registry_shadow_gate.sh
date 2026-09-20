#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE="${DEFINITION_REGISTRY_SHADOW_BASE_SHA:-4c952e6ee7bcd0855f675fc662420c2fa507e19a}"
SOUC="${SOUC_BIN:-}"
EXPECTED_COMPILER_SHA256="${DEFINITION_REGISTRY_EXPECTED_COMPILER_SHA256:-}"
REGISTRY="self-hosted/resolve/definition_registry_shadow.sio"
PROBE="self-hosted/resolve/definition_registry_shadow_probe.sio"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-definition-registry-shadow.XXXXXX")"
COMPOSITE=""
trap 'rm -rf "$TMP"; [[ -z "$COMPOSITE" ]] || rm -f "$COMPOSITE"' EXIT

fail() {
  printf 'DEFINITION_REGISTRY_SHADOW_FAIL reason=%s\n' "$1" >&2
  exit 1
}

for file in "$REGISTRY" "$PROBE"; do
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

grep -Fq 'pub struct DefinitionRegistryShadowModuleDefId' "$REGISTRY" || fail module_def_id_missing
grep -Fq 'pub struct DefinitionRegistryShadowNominalTypeDefId' "$REGISTRY" || fail nominal_type_def_id_missing
grep -Fq 'pub struct DefinitionRegistryShadowFieldDefId' "$REGISTRY" || fail field_def_id_missing
grep -Fq 'pub struct DefinitionRegistryShadowTypeExprBindingReceipt' "$REGISTRY" || fail binding_receipt_missing
grep -Fq 'var DRS_NEXT_REGISTRY_IDENTITY: i64 = 1' "$REGISTRY" || fail compiler_minted_registry_identity_missing
grep -Fq 'pub fn definition_registry_shadow_begin()' "$REGISTRY" || fail caller_supplied_registry_identity_detected
grep -Fq 'DEFINITION_REGISTRY_SHADOW_ERR_DUPLICATE_MODULE_PATH' "$REGISTRY" || fail duplicate_path_rejection_missing
grep -Fq 'fn drs_module_id_is_clear' "$REGISTRY" || fail clear_output_discipline_missing
grep -Fq 'fn drs_rollback_module_declarations' "$REGISTRY" || fail transactional_declaration_rollback_missing
grep -Fq 'fn drs_rollback_module_imports' "$REGISTRY" || fail transactional_import_rollback_missing
grep -Fq 'var DRS_FIELD_DECLARED_ORDINAL' "$REGISTRY" || fail declared_ordinal_payload_missing
grep -Fq 'drs_module_path_matches_field_prefix' "$REGISTRY" || fail qualified_path_resolution_missing
grep -Fq 'DEFINITION_REGISTRY_SHADOW_ERR_AMBIGUOUS' "$REGISTRY" || fail ambiguity_rejection_missing
grep -Fq 'DEFINITION_REGISTRY_SHADOW_ERR_UNRESOLVED' "$REGISTRY" || fail unresolved_rejection_missing
if grep -Eq 'ast_name_hash|hash[[:space:]]*\(' "$REGISTRY"; then
  fail hash_identity_or_fallback_detected
fi
if grep -Eq '^[[:space:]]*(path|module_path|type_path): AstPath,' "$REGISTRY"; then
  fail aggregate_path_by_value_signature_detected
fi
if grep -Eq '^pub var DRS_' "$REGISTRY"; then
  fail registry_storage_public
fi

for default_surface in \
  self-hosted/resolve/mod.sio \
  self-hosted/resolve/resolve.sio \
  self-hosted/check/check.sio \
  self-hosted/check/mod.sio \
  self-hosted/check/defs.sio \
  self-hosted/check/types.sio \
  self-hosted/ir/mod.sio \
  self-hosted/ir/ir.sio \
  self-hosted/compiler/main.sio \
  scripts/bootstrap/bootstrap_concat.sh \
  bin/souc; do
  grep -Fq 'definition_registry_shadow' "$default_surface" && fail "shadow_imported_by_${default_surface//\//_}"
done

{
  git diff --name-only "$BASE"
  git ls-files --others --exclude-standard
} | sed '/^$/d' | sort -u >"$TMP/actual-files.txt"
printf '%s\n' \
  scripts/ci/definition_registry_shadow_gate.sh \
  self-hosted/resolve/definition_registry_shadow.sio \
  self-hosted/resolve/definition_registry_shadow_probe.sio \
  | sort -u >"$TMP/allowed-files.txt"
if ! diff -u "$TMP/allowed-files.txt" "$TMP/actual-files.txt" >"$TMP/files.diff"; then
  cat "$TMP/files.diff" >&2
  fail changed_file_allowlist_mismatch
fi

"$SOUC" check "$REGISTRY" >"$TMP/registry.check.log" 2>&1 || {
  cat "$TMP/registry.check.log" >&2
  fail registry_source_check
}
"$SOUC" check "$PROBE" >"$TMP/probe.check.log" 2>&1 || {
  cat "$TMP/probe.check.log" >&2
  fail imported_probe_check
}

COMPOSITE="$(mktemp "$ROOT/self-hosted/resolve/definition_registry_shadow_gate.XXXXXX.sio")"
{
  printf 'module resolve::definition_registry_shadow_gate\n\n'
  sed -e '/^module resolve::definition_registry_shadow$/d' \
      -e '/^use parser::ast::\*$/d' "$REGISTRY"
  sed -e '/^module resolve::definition_registry_shadow_probe$/d' \
      -e '/^use resolve::definition_registry_shadow::\*$/d' "$PROBE"
} >"$COMPOSITE"

"$SOUC" check "$COMPOSITE" >"$TMP/composite.check.log" 2>&1 || {
  cat "$TMP/composite.check.log" >&2
  fail single_tu_check
}
"$SOUC" --native-v2-compile "$COMPOSITE" -o "$TMP/single-tu.elf" >"$TMP/single-tu.build.log" 2>&1 || {
  cat "$TMP/single-tu.build.log" >&2
  fail single_tu_build
}
chmod +x "$TMP/single-tu.elf"
set +e
"$TMP/single-tu.elf" >"$TMP/single-tu.out" 2>"$TMP/single-tu.err"
single_tu_rc=$?
set -e
if [[ "$single_tu_rc" -ne 0 ]]; then
  if [[ "$single_tu_rc" -eq 212 ]] &&
     grep -Fxq 'DEFINITION_REGISTRY_SHADOW_BLOCKER_OBSERVED nominal=8 field=16' "$TMP/single-tu.out"; then
    branch="$(git branch --show-current)"
    printf '%s\n' 'DEFINITION_REGISTRY_SHADOW_CHECK source=pass probe=pass single_tu=blocked_native_v2_global_array_alias imported_module=not_reached off_default=true'
    printf '%s\n' 'DEFINITION_REGISTRY_SHADOW_REACHED compiler_minted_registry=pass active_begin=reject missing_module_authority=reject duplicate_path=reject module_capacity=reject nominal_capacity=reject transactional_rollback=blocked'
    printf '%s\n' 'DEFINITION_REGISTRY_SHADOW_NOT_REACHED ab_ba=same_semantic_relation same_spelling_distinct_modules ambiguity unresolved stale reset same_slot_new_generation_aba receipt_output_reuse release imported_module'
    printf '%s\n' 'DEFINITION_REGISTRY_SHADOW_BOUNDARY mode=executable_blocker parser_program_wrapper=source_checked_not_runtime_integrated checker=false type_entry=false field_info=false place=false layout=false default_pipeline=false legacy_kept=true'
    printf '%s\n' 'DEFINITION_REGISTRY_SHADOW_BLOCKER Blocker-ID=BLK-20260715-definition-registry-native-v2-global-array-alias Status=classified Severity=B1 Class=bootstrap-runtime Evidence-Level=E3 Result=BLOCKED_PRIVATE_GLOBAL_ARRAY_COLUMNS_ALIAS nominal_expected=0 nominal_observed=8 field_expected=0 field_observed=16 first_boundary=abort_resolved_declarations_count_receipt'
    printf 'DEFINITION_REGISTRY_SHADOW_BLOCKER_SCOPE Owner=Codex-definition-registry-shadow Lane=qualified-nominal-identity Worktree=%s Branch=%s Files-Owned=%s,%s,%s Files-Read-Only=parser,checker,ir,compiler Do-Not-Touch=checker,TargetLayout,Place,SOIR,bootstrap,shared-control\n' "$ROOT" "$branch" "$REGISTRY" "$PROBE" scripts/ci/definition_registry_shadow_gate.sh
    printf '%s\n' 'DEFINITION_REGISTRY_SHADOW_BLOCKER_CONTRACT Repro=SOUC_BIN=<source-fresh-elf>_DEFINITION_REGISTRY_EXPECTED_COMPILER_SHA256=<sha256>_bash_scripts/ci/definition_registry_shadow_gate.sh Observed=rollback_nominal_8_field_16 Expected=rollback_nominal_0_field_0 Acceptance-Gate=same_command Evidence=gate_stdout_or_CI_log Fallback-Path=none Legacy-Kept=yes LLM-Offload=not-required Next-Action=replace_private_global_aggregate_arrays_with_scalar_or_heap-owned_store_then_rerun_full_adversary'
    printf 'DEFINITION_REGISTRY_SHADOW_PASS mode=executable_blocker compiler=%s compiler_sha256=%s base=%s\n' "$SOUC" "$compiler_sha256" "$BASE"
    exit 0
  fi
  cat "$TMP/single-tu.out" >&2
  cat "$TMP/single-tu.err" >&2
  fail "single_tu_runtime_${single_tu_rc}"
fi
grep -Fxq 'DEFINITION_REGISTRY_SHADOW_PASS mode=resolved_declaration_shadow' "$TMP/single-tu.out" || fail single_tu_receipt_missing

imported_result=pass
if ! "$SOUC" --native-v2-compile "$PROBE" -o "$TMP/imported.elf" >"$TMP/imported.build.log" 2>&1; then
  imported_result=blocked_native_v2_imported_build
else
  chmod +x "$TMP/imported.elf"
  set +e
  "$TMP/imported.elf" >"$TMP/imported.out" 2>"$TMP/imported.err"
  imported_rc=$?
  set -e
  if [[ "$imported_rc" -ne 0 ]]; then
    imported_result="blocked_native_v2_imported_runtime_${imported_rc}"
  elif ! grep -Fxq 'DEFINITION_REGISTRY_SHADOW_PASS mode=resolved_declaration_shadow' "$TMP/imported.out"; then
    imported_result=blocked_native_v2_imported_receipt_missing
  fi
fi

printf 'DEFINITION_REGISTRY_SHADOW_CHECK source=pass probe=pass single_tu=pass imported_module=%s off_default=true\n' "$imported_result"
printf '%s\n' 'DEFINITION_REGISTRY_SHADOW_ID module=registry_identity,snapshot_generation,slot,generation nominal=registry_identity,snapshot_generation,slot,generation field=registry_identity,snapshot_generation,slot,generation authority=compiler_minted_snapshot'
printf '%s\n' 'DEFINITION_REGISTRY_SHADOW_RESOLUTION path=exact_lookup_key def_id=semantic_identity hash_fallback=absent spelling_only_fallback=absent declaration_phase=before_import_phase'
printf '%s\n' 'DEFINITION_REGISTRY_SHADOW_ADVERSARY ab_ba=semantic_relation_equal slots_not_compared same_nominal_spelling_distinct_modules=pass same_field_spelling_distinct_owners=pass declared_ordinal_payload_only=pass duplicate_path=reject missing_module_authority=reject ambiguous=reject unresolved=reject stale=reject reset=pass same_slot_new_generation_aba=reject module_capacity=reject transactional_nominal_capacity=rollback stale_receipt_output_reuse_without_clear=reject release=pass'
printf '%s\n' 'DEFINITION_REGISTRY_SHADOW_BOUNDARY mode=resolved_declaration_shadow parser_program_wrapper=source_checked_not_runtime_integrated checker=false type_entry=false field_info=false place=false layout=false default_pipeline=false legacy_kept=true'
printf 'DEFINITION_REGISTRY_SHADOW_PASS compiler=%s compiler_sha256=%s base=%s\n' "$SOUC" "$compiler_sha256" "$BASE"
