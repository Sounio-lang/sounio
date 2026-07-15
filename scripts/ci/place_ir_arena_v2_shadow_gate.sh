#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE="${PLACE_IR_ARENA_V2_BASE_SHA:-8f6e88e5260f55e4db3e7cefa49e1cf4ecc73634}"
SOUC="${SOUC_BIN:-$ROOT/bin/souc}"
ARENA="self-hosted/ir/arena_v2_shadow.sio"
PLACE="self-hosted/ir/arena_v2_place_shadow.sio"
WITNESS="tests/native-v2/place_ir_arena_v2_shadow_witness.sio"
IMPORT_PROBE="self-hosted/ir/arena_v2_place_import_probe.sio"
LEGACY_WITNESS="tests/native-v2/place_ir_legacy_nominal_collision_witness.sio"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-place-ir-arena-v2.XXXXXX")"
COMPOSITE=""

cleanup() {
  rm -rf "$TMP"
  [[ -z "$COMPOSITE" ]] || rm -f "$COMPOSITE"
}
trap cleanup EXIT

fail() {
  printf 'PLACE_IR_ARENA_V2_FAIL reason=%s\n' "$1" >&2
  exit 1
}

for file in "$ARENA" "$PLACE" "$WITNESS" "$IMPORT_PROBE" "$LEGACY_WITNESS"; do
  [[ -f "$file" ]] || fail "missing_${file//\//_}"
done
[[ -x "$SOUC" ]] || fail compiler_missing
git cat-file -e "$BASE^{commit}" 2>/dev/null || fail base_sha_unavailable

grep -Fq 'pub struct IrModuleArenaV2PlaceId {' "$PLACE" || fail place_id_missing
grep -Fq 'pub let IR_MODULE_ARENA_V2_PLACE_CAPACITY: i64 = 2' "$PLACE" || fail place_capacity_not_two
grep -Fq 'pub let IR_MODULE_ARENA_V2_PLACE_PROJECTION_CAPACITY: i64 = 2' "$PLACE" || fail projection_capacity_not_two
grep -Fq 'out: &! IrModuleArenaV2PlaceId' "$PLACE" || fail place_allocation_not_out_parameter
grep -Fq 'ir_module_arena_v2_module_id_generation(module)' "$PLACE" || fail module_generation_binding_missing
grep -Fq 'ir_module_arena_v2_module_mutation_epoch(module)' "$PLACE" || fail module_epoch_binding_missing
grep -Fq 'proj_field_ordinal(cell_a) == proj_field_ordinal(cell_b)' "$PLACE" || fail structural_field_ordinal_missing
grep -Fq 'proj_owner_type(cell_a) == proj_owner_type(cell_b)' "$PLACE" || fail structural_owner_type_missing
grep -Fq 'proj_result_type(cell_a) == proj_result_type(cell_b)' "$PLACE" || fail structural_result_type_missing
grep -Fq 'IR_MODULE_ARENA_V2_PLACE_ERR_SAME_HASH_STRUCTURAL_COLLISION' "$PLACE" || fail same_hash_collision_status_missing
grep -Fq 'ir_module_arena_v2_place_record_store_after_load' "$PLACE" || fail ordered_store_missing
grep -Fq 'ir_module_arena_v2_place_finalize' "$PLACE" || fail place_finalize_missing
grep -Fq 'ir_module_arena_v2_place_discard' "$PLACE" || fail stale_lifecycle_discard_missing
grep -Fq 'place_structural_witness_not_parity' "$WITNESS" || fail nonparity_classification_missing
grep -Fq 'PLACE_IR_ARENA_V2_IMPORT_PASS' "$IMPORT_PROBE" || fail import_probe_marker_missing
grep -Fq 'if typed.shared != 42' "$LEGACY_WITNESS" || fail typed_legacy_control_missing
grep -Fq 'make_b().shared' "$LEGACY_WITNESS" || fail legacy_collision_expression_missing
grep -Fq 'PRs #894 and #910' "$PLACE" || fail characterization_reuse_attribution_missing

if grep -Eq '\[[[:space:]]*[A-Za-z0-9_:]+[[:space:]]*;[[:space:]]*[0-9]+' "$PLACE"; then
  fail aggregate_array_storage_present
fi
if grep -Eq '^pub var PLACE_|^pub var PROJ_' "$PLACE"; then
  fail scalar_columns_public
fi
if grep -Fq 'pub struct PlaceProjection' "$PLACE" || grep -Fq 'pub struct PlaceStore' "$PLACE"; then
  fail aggregate_place_or_projection_store_present
fi
if grep -Eq -- '->[[:space:]]*(Place|IrModuleArenaV2Place)([[:space:]<{]|$)' "$PLACE"; then
  fail aggregate_place_return_present
fi

for default_surface in \
  self-hosted/ir/mod.sio \
  self-hosted/ir/lower.sio \
  self-hosted/ir/ir.sio \
  self-hosted/ir/arena_v2_shadow.sio \
  self-hosted/ir/arena_v2_soir_bridge.sio \
  self-hosted/ir/soir_writer.sio \
  self-hosted/ir/soir_core.sio \
  self-hosted/ir/serialize.sio \
  self-hosted/ir/heap_storage.sio \
  self-hosted/parser/ast.sio \
  self-hosted/parser/items.sio \
  self-hosted/compiler/main.sio \
  self-hosted/native/codegen_x86_linux.sio \
  self-hosted/native/runtime_context.sio \
  scripts/ci/madaros_global_init_transport_gate.sh \
  scripts/bootstrap/bootstrap_concat.sh; do
  grep -Fq 'arena_v2_place_shadow' "$default_surface" && fail "place_imported_by_${default_surface//\//_}"
done

if ! git diff --quiet "$BASE" -- \
    self-hosted/ir/mod.sio \
    self-hosted/ir/lower.sio \
    self-hosted/ir/ir.sio \
    self-hosted/ir/arena_v2_shadow.sio \
    self-hosted/ir/arena_v2_soir_bridge.sio \
    self-hosted/ir/soir_writer.sio \
    self-hosted/ir/soir_core.sio \
    self-hosted/ir/serialize.sio \
    self-hosted/ir/heap_storage.sio \
    self-hosted/parser/ast.sio \
    self-hosted/parser/items.sio \
    self-hosted/compiler/main.sio \
    self-hosted/native/codegen_x86_linux.sio \
    self-hosted/native/runtime_context.sio \
    scripts/ci/madaros_global_init_transport_gate.sh \
    scripts/bootstrap/bootstrap_concat.sh; then
  fail legacy_or_default_surface_changed
fi

"$SOUC" check "$PLACE" >"$TMP/place.check.log" 2>&1 || {
  cat "$TMP/place.check.log" >&2
  fail place_source_check
}

COMPOSITE="$(mktemp "$ROOT/self-hosted/ir/place_ir_arena_v2_shadow_gate.XXXXXX.sio")"
{
  printf 'module ir::place_ir_arena_v2_shadow_gate\n\n'
  sed '/^module ir::arena_v2_shadow$/d' "$ARENA"
  sed -e '/^module ir::arena_v2_place_shadow$/d' \
      -e '/^use ir::arena_v2_shadow::\*$/d' "$PLACE"
  sed -e '/^use ir::arena_v2_shadow::\*$/d' \
      -e '/^use ir::arena_v2_place_shadow::\*$/d' "$WITNESS"
} >"$COMPOSITE"

"$SOUC" check "$COMPOSITE" >"$TMP/composite.log" 2>&1 || {
  cat "$TMP/composite.log" >&2
  fail composite_check
}
"$SOUC" --native-v2-compile "$COMPOSITE" -o "$TMP/place-shadow.elf" >>"$TMP/composite.log" 2>&1 || {
  cat "$TMP/composite.log" >&2
  fail composite_build
}
chmod +x "$TMP/place-shadow.elf"
"$TMP/place-shadow.elf" >"$TMP/place-shadow.out" 2>&1 || {
  cat "$TMP/place-shadow.out" >&2
  fail composite_runtime
}
grep -Fxq 'PLACE_IR_ARENA_V2_SHADOW_PASS mode=place_structural_witness_not_parity' "$TMP/place-shadow.out" || {
  cat "$TMP/place-shadow.out" >&2
  fail shadow_receipt_missing
}

# This probe is intentionally not concatenated. It exercises imported calls
# carrying ModuleId, PlaceId, scalar projection payloads, and access arguments.
"$SOUC" --native-v2-compile "$IMPORT_PROBE" -o "$TMP/import-probe.elf" >"$TMP/import-probe.log" 2>&1 || {
  cat "$TMP/import-probe.log" >&2
  fail imported_probe_build
}
chmod +x "$TMP/import-probe.elf"
"$TMP/import-probe.elf" >"$TMP/import-probe.out" 2>&1 || {
  cat "$TMP/import-probe.out" >&2
  fail imported_probe_runtime
}
grep -Fxq 'PLACE_IR_ARENA_V2_IMPORT_PASS' "$TMP/import-probe.out" || {
  cat "$TMP/import-probe.out" >&2
  fail imported_probe_receipt_missing
}

"$SOUC" compile "$LEGACY_WITNESS" -o "$TMP/legacy.elf" >"$TMP/legacy.log" 2>&1 || {
  cat "$TMP/legacy.log" >&2
  fail legacy_compile
}
chmod +x "$TMP/legacy.elf"
set +e
"$TMP/legacy.elf" >"$TMP/legacy.out" 2>&1
legacy_rc=$?
set -e
case "$legacy_rc" in
  11) legacy_observation=current_name_fallback_counterexample ;;
  42) legacy_observation=legacy_fixed ;;
  *)
    cat "$TMP/legacy.out" >&2
    fail "legacy_runtime_unexpected_${legacy_rc}"
    ;;
esac

compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
printf '%s\n' 'PLACE_IR_ARENA_V2_CHECK source=pass composite=pass imported_api_probe=pass default_surfaces=unchanged off_default=true'
printf '%s\n' 'PLACE_IR_ARENA_V2_ID place_id=slot,generation lifecycle=build,finalized,released stale_discard=generation_only root_binding=module_arena,module_slot,module_generation,module_mutation_epoch root=identity,type,layout'
printf '%s\n' 'PLACE_IR_ARENA_V2_PROJECTION storage=private_scalar_columns order=append declared_identity=module,type,layout,ordinal result=type,layout authority=caller_supplied_shadow name_hash=diagnostic_only nested_type_chain=self_consistency_checked'
printf '%s\n' 'PLACE_IR_ARENA_V2_ADVERSARY same_hash_structural_collision=reject nested=pass unfinalized_access=reject append_after_finalize=reject load_then_store=pass write_flag_zero=reject write_authority=caller_supplied_shadow module_stale=reject module_reuse=reject stale_discard=pass stale_capacity_recovery=pass place_aba=reject structural_distinct=pass place_capacity=distinct projection_capacity=distinct'
printf 'PLACE_IR_ARENA_V2_DIFFERENTIAL mode=place_structural_witness_not_parity shadow_declared_field_ordinal=1 legacy_typed_control=42 legacy_call_expression_rc=%s legacy_observation=%s\n' "$legacy_rc" "$legacy_observation"
printf '%s\n' 'PLACE_IR_ARENA_V2_BOUNDARY canonical=false backend=false writer=false serializer=false alias_analysis=false canonical_type_layout_resolution=false structural_matrix=composite imported_api=smoke legacy_kept=true'
printf 'PLACE_IR_ARENA_V2_PASS compiler=%s compiler_sha256=%s base=%s\n' "$SOUC" "$compiler_sha256" "$BASE"
