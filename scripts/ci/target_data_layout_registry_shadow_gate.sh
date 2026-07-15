#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE="${TARGET_DATA_LAYOUT_REGISTRY_SHADOW_BASE_SHA:-4c952e6ee7bcd0855f675fc662420c2fa507e19a}"
SOUC="${SOUC_BIN:-}"
EXPECTED_COMPILER_SHA256="${TARGET_DATA_LAYOUT_EXPECTED_COMPILER_SHA256:-}"
REGISTRY="self-hosted/native/target_data_layout_registry_shadow.sio"
PROBE="self-hosted/native/target_data_layout_registry_shadow_probe.sio"
TARGET_POLICY="self-hosted/native/target_policy.sio"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-target-data-layout-registry-shadow.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

fail() {
  printf 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_FAIL reason=%s\n' "$1" >&2
  exit 1
}

for file in "$REGISTRY" "$PROBE" "$TARGET_POLICY"; do
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

grep -Fq 'Semantic-Lane-ID: SOUNIO-TARGET-DATA-LAYOUT-AUTHORITY-SHADOW' "$REGISTRY" || fail semantic_lane_missing
grep -Fq 'pub struct TargetDataLayoutId {' "$REGISTRY" || fail generational_id_missing
grep -Fq 'store_identity: i64' "$REGISTRY" || fail store_identity_missing
grep -Fq 'generation: i64' "$REGISTRY" || fail generation_missing
grep -Fq 'out: &! TargetDataLayoutId' "$REGISTRY" || fail allocation_not_out_parameter
grep -Fq 'use native::target_policy::*' "$REGISTRY" || fail canonical_target_policy_import_missing
grep -Fq 'native_policy_arch_x86_64()' "$REGISTRY" || fail canonical_x86_identity_missing
grep -Fq 'native_policy_arch_aarch64()' "$REGISTRY" || fail canonical_aarch64_identity_missing
grep -Fq 'native_policy_os_linux()' "$REGISTRY" || fail canonical_linux_identity_missing
grep -Fq 'native_policy_os_macos()' "$REGISTRY" || fail canonical_macos_identity_missing
grep -Fq 'target_data_layout_profile_x86_64_macos_darwin_v1()' "$REGISTRY" || fail darwin_profile_missing
grep -Fq 'target_os_abi_darwin_amd64_v1()' "$REGISTRY" || fail darwin_abi_policy_missing
grep -Fq 'target_os_abi_aapcs64_elf_v1()' "$REGISTRY" || fail aapcs64_elf_policy_missing
grep -Fq 'target_data_layout_slot_profile_integrity_code((*id).slot) != 0' "$REGISTRY" || fail registered_profile_validation_missing
grep -Fq 'pub fn target_data_layout_registry_shadow_store_init()' "$REGISTRY" || fail compiler_minted_store_init_missing
grep -Fq 'pub fn target_data_layout_registry_shadow_store_reset()' "$REGISTRY" || fail compiler_minted_store_reset_missing
grep -Fq 'target_wide_float_abi_undeclared()' "$REGISTRY" || fail wide_float_undeclared_policy_missing
grep -Fq 'target_data_layout_registry_shadow_f128_abi_status' "$REGISTRY" || fail f128_status_missing
grep -Fq 'target_data_layout_registry_shadow_f256_abi_status' "$REGISTRY" || fail f256_status_missing
grep -Fq 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_ERR_WIDE_FLOAT_UNSUPPORTED' "$REGISTRY" || fail wide_float_unsupported_status_missing
grep -Fq 'target_data_layout_registry_shadow_declared_storage_policy_equal' "$PROBE" || fail coincident_storage_adversary_missing
grep -Fq 'target_data_layout_registry_shadow_abi_policy_equal' "$PROBE" || fail distinct_abi_adversary_missing
grep -Fq 'target_data_layout_registry_shadow_arch_id(&reused) != native_policy_arch_aarch64()' "$PROBE" || fail aarch64_live_profile_validation_missing
grep -Fq 'Registered-profile traversal' "$PROBE" || fail registered_profile_traversal_missing
grep -Fq 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_ERR_STALE_ID' "$PROBE" || fail stale_adversary_missing
grep -Fq 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_ERR_CAPACITY' "$PROBE" || fail capacity_adversary_missing
grep -Fq 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_ERR_DUPLICATE_PROFILE' "$PROBE" || fail duplicate_adversary_missing
grep -Fq 'classification=target_authority_shadow_not_field_layout' "$PROBE" || fail boundary_classification_missing

if grep -Eq '\[[[:space:]]*[A-Za-z0-9_:]+[[:space:]]*;[[:space:]]*[0-9]+' "$REGISTRY"; then
  fail aggregate_array_storage_present
fi
if grep -Eq '^pub var TARGET_DATA_LAYOUT_' "$REGISTRY"; then
  fail scalar_columns_public
fi
if grep -Eq 'hash|fingerprint' "$REGISTRY"; then
  fail hash_or_fingerprint_authority_present
fi
if grep -Eq 'FieldLayoutReceipt|field_idx|field_ordinal|byte_offset' "$REGISTRY" "$PROBE"; then
  fail field_layout_authority_present
fi
if grep -Eq 'f128_(size|alignment|storage|bytes)|f256_(size|alignment|storage|bytes)' "$REGISTRY" "$PROBE"; then
  fail implicit_wide_float_physical_policy_present
fi
if grep -Eq 'TARGET_DATA_LAYOUT_WIDE_FLOAT_ABI_POLICY_[01][[:space:]]*=[[:space:]]*(8|16|32)' "$REGISTRY"; then
  fail implicit_wide_float_byte_default_present
fi
if grep -Eq '^pub fn target_data_layout_registry_shadow_(set|write|alloc_raw|alloc_policy)' "$REGISTRY"; then
  fail caller_supplied_policy_mutator_present
fi
if grep -Fq 'store_init(store_identity' "$REGISTRY" "$PROBE"; then
  fail caller_minted_store_identity_present
fi

for default_surface in \
  self-hosted/native/mod.sio \
  self-hosted/native/target_policy.sio \
  self-hosted/native/contract.sio \
  self-hosted/native/abi_lower.sio \
  self-hosted/native/codegen_x86_linux.sio \
  self-hosted/check/check.sio \
  self-hosted/check/mod.sio \
  self-hosted/ir/mod.sio \
  self-hosted/ir/arena_v2_shadow.sio \
  self-hosted/ir/arena_v2_place_shadow.sio \
  self-hosted/compiler/main.sio \
  scripts/lib/resolve_souc.sh \
  scripts/lib/resolve_madaros.sh \
  scripts/ci/build_native_souc.sh \
  .github/workflows/ci.yml; do
  grep -Fq 'target_data_layout_registry_shadow' "$default_surface" && fail "shadow_imported_by_${default_surface//\//_}"
done

if ! git diff --quiet "$BASE" -- \
    self-hosted/native/mod.sio \
    self-hosted/native/target_policy.sio \
    self-hosted/native/contract.sio \
    self-hosted/native/abi_lower.sio \
    self-hosted/native/codegen_x86_linux.sio \
    self-hosted/check/check.sio \
    self-hosted/check/mod.sio \
    self-hosted/ir/mod.sio \
    self-hosted/ir/arena_v2_shadow.sio \
    self-hosted/ir/arena_v2_place_shadow.sio \
    self-hosted/compiler/main.sio \
    scripts/lib/resolve_souc.sh \
    scripts/lib/resolve_madaros.sh \
    scripts/ci/build_native_souc.sh \
    .github/workflows/ci.yml; then
  fail default_or_owned_surface_changed
fi

while IFS= read -r changed; do
  [[ -z "$changed" ]] && continue
  case "$changed" in
    "$REGISTRY"|"$PROBE"|scripts/ci/target_data_layout_registry_shadow_gate.sh) ;;
    *) fail "unexpected_changed_file_${changed//\//_}" ;;
  esac
done < <(git diff --name-only "$BASE")

run_single_tu_diagnostic() {
  local composite="$TMP/single-tu-diagnostic.sio"
  {
    printf 'module native::target_data_layout_registry_shadow_single_tu_diagnostic\n\n'
    sed '/^module native::target_policy$/d' "$TARGET_POLICY"
    sed -e '/^module native::target_data_layout_registry_shadow$/d' \
        -e '/^use native::target_policy::\*$/d' "$REGISTRY"
    sed -e '/^use native::target_data_layout_registry_shadow::\*$/d' \
        -e '/^use native::target_policy::\*$/d' "$PROBE"
  } >"$composite"

  if ! "$SOUC" --native-v2-compile "$composite" -o "$TMP/single-tu.elf" >"$TMP/single-tu.build.log" 2>&1; then
    printf '%s\n' 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_DIAGNOSTIC single_tu=build_failed authority=diagnostic_only' >&2
    return
  fi
  chmod +x "$TMP/single-tu.elf"
  set +e
  "$TMP/single-tu.elf" >"$TMP/single-tu.out" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 ]] && grep -Fxq 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_PASS classification=target_authority_shadow_not_field_layout' "$TMP/single-tu.out"; then
    printf '%s\n' 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_DIAGNOSTIC single_tu=pass authority=diagnostic_only' >&2
  else
    cat "$TMP/single-tu.out" >&2
    printf 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_DIAGNOSTIC single_tu=runtime_failed rc=%s authority=diagnostic_only\n' "$rc" >&2
  fi
}

"$SOUC" check "$REGISTRY" >"$TMP/registry.check.log" 2>&1 || {
  cat "$TMP/registry.check.log" >&2
  fail registry_source_check
}
"$SOUC" check "$PROBE" >"$TMP/probe.check.log" 2>&1 || {
  cat "$TMP/probe.check.log" >&2
  fail probe_source_check
}
"$SOUC" --native-v2-compile "$PROBE" -o "$TMP/probe.elf" >"$TMP/probe.build.log" 2>&1 || {
  cat "$TMP/probe.build.log" >&2
  fail probe_native_build
}
probe_magic="$(od -An -tx1 -N4 "$TMP/probe.elf" | tr -d ' \n')"
[[ "$probe_magic" == "7f454c46" ]] || fail probe_output_not_elf
chmod +x "$TMP/probe.elf"
"$TMP/probe.elf" >"$TMP/probe.out" 2>&1 || {
  run_single_tu_diagnostic
  cat "$TMP/probe.out" >&2
  fail probe_runtime
}
grep -Fxq 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_PASS classification=target_authority_shadow_not_field_layout' "$TMP/probe.out" || {
  cat "$TMP/probe.out" >&2
  fail probe_receipt_missing
}

printf '%s\n' 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_CHECK source=pass imported_probe=pass default_surfaces=unchanged off_default=true compiler=source_fresh_sha_pinned output=elf'
printf '%s\n' 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_AUTHORITY identity=compiler_minted_store_identity,slot,generation reset=monotonic profile_input=closed_registered_id canonical_arch_os=native_target_policy storage=private_scalar_columns hash_authority=false'
printf '%s\n' 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_POLICY fields=profile,arch,os,os_abi,endianness,pointer_size,pointer_alignment,scalar_alignment_policy,vector_alignment_policy,wide_float_abi_policy,managed_object_layout_version,layout_algorithm_version equal_declared_storage_distinct_target=pass abi_policy_distinct=pass aarch64_live_profile=pass'
printf '%s\n' 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_WIDE_FLOAT policy=undeclared f128=unsupported f256=unsupported implicit_physical_default=absent narrowing=absent'
printf '%s\n' 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_ADVERSARY stale=reject reset=reject aba=reject capacity=reject duplicate_profile=reject unknown_profile=reject registered_profile_traversal=pass'
printf '%s\n' 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_BOUNDARY classification=target_authority_shadow_not_field_layout field_layout_receipt=false place_binder_ready=false backend=false complete_abi=false legacy_kept=true'
printf 'TARGET_DATA_LAYOUT_REGISTRY_SHADOW_PASS compiler=%s compiler_sha256=%s base=%s\n' "$SOUC" "$compiler_sha256" "$BASE"
