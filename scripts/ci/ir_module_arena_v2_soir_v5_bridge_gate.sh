#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE="${IR_MODULE_ARENA_V2_SOIR_BASE_SHA:-6f83acb9d9331c5f6312e63fadf3fd87eb9940ba}"
SOUC="${SOUC_BIN:-$ROOT/bin/souc}"
ARENA="self-hosted/ir/arena_v2_shadow.sio"
WRITER="self-hosted/ir/soir_writer.sio"
CORE="self-hosted/ir/soir_core.sio"
BRIDGE="self-hosted/ir/arena_v2_soir_bridge.sio"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-ir-module-arena-v2-soir-v5.XXXXXX")"
COMPOSITES=()

cleanup() {
  rm -rf "$TMP"
  if [[ "${#COMPOSITES[@]}" -gt 0 ]]; then rm -f "${COMPOSITES[@]}"; fi
}
trap cleanup EXIT

fail() {
  printf 'IR_MODULE_ARENA_V2_SOIR_V5_FAIL reason=%s\n' "$1" >&2
  exit 1
}

for file in "$ARENA" "$WRITER" "$CORE" "$BRIDGE"; do
  [[ -f "$file" ]] || fail "missing_${file//\//_}"
done
[[ -x "$SOUC" ]] || fail compiler_missing
git cat-file -e "$BASE^{commit}" 2>/dev/null || fail base_sha_unavailable

grep -Fq 'pub struct IrModuleArenaV2SoirPlanId {' "$BRIDGE" || fail plan_id_missing
grep -Fq 'pub let IR_MODULE_ARENA_V2_SOIR_PLAN_CAPACITY: i64 = 2' "$BRIDGE" || fail plan_capacity_not_two
grep -Fq 'pub fn ir_module_arena_v2_soir_preflight_empty_v5(' "$BRIDGE" || fail preflight_missing
grep -Fq 'out: &! IrModuleArenaV2SoirPlanId' "$BRIDGE" || fail preflight_not_out_parameter
grep -Fq 'pub fn ir_module_arena_v2_soir_emit_empty_v5(' "$BRIDGE" || fail emit_missing
grep -Fq 'plan_id: &IrModuleArenaV2SoirPlanId' "$BRIDGE" || fail emit_not_plan_id
grep -Fq 'soir_writer_preflight_scalar_empty_module_v5' "$BRIDGE" || fail writer_preflight_not_delegated
grep -Fq 'soir_writer_emit_scalar_empty_module_v5' "$BRIDGE" || fail writer_emit_not_delegated
grep -Fq 'SOIR_WRITER_PLAN_CORE_BEGIN' "$WRITER" || fail writer_plan_marker_missing
grep -Fq 'SOIR_WRITER_SHARED_SCALAR_PRIMITIVES_BEGIN' "$WRITER" || fail writer_shared_primitives_marker_missing
grep -Fq 'SOIR_WRITER_SCALAR_EMPTY_CORE_BEGIN' "$WRITER" || fail writer_scalar_marker_missing
grep -Fq 'pub fn soir_writer_preflight_empty_extensions_v5(' "$WRITER" || fail legacy_preflight_removed
grep -Fq 'pub fn soir_writer_emit_empty_extensions_v5(' "$WRITER" || fail legacy_emit_removed

if grep -Eq '(->|:)[[:space:]]*IrModule([[:space:]>,{]|$)' "$BRIDGE" || grep -Fq 'ir_empty_module(' "$BRIDGE"; then
  fail legacy_irmodule_value_boundary
fi
if grep -Eq '(out_buf|buf):[[:space:]]*\[i8;[[:space:]]*131072\]' "$BRIDGE"; then
  fail buffer_passed_by_value
fi
if grep -Eq '^pub var IR_MODULE_ARENA_V2_SOIR_PLAN_' "$BRIDGE"; then
  fail plan_columns_public
fi
if grep -Fq 'pub struct IrModuleArenaV2SoirPlan {' "$BRIDGE"; then
  fail aggregate_plan_crosses_bridge
fi

for default_surface in \
  self-hosted/ir/serialize.sio \
  self-hosted/ir/soir_core.sio \
  self-hosted/ir/heap_storage.sio \
  self-hosted/ir/mod.sio \
  self-hosted/ir/lower.sio \
  self-hosted/parser/ast.sio \
  self-hosted/parser/items.sio \
  self-hosted/compiler/main.sio \
  self-hosted/native/codegen_x86_linux.sio \
  self-hosted/native/runtime_context.sio \
  scripts/ci/madaros_global_init_transport_gate.sh \
  scripts/bootstrap/bootstrap_concat.sh; do
  grep -Fq 'arena_v2_soir_bridge' "$default_surface" && fail "bridge_imported_by_${default_surface//\//_}"
done

if ! git diff --quiet "$BASE" -- \
    self-hosted/ir/serialize.sio \
    self-hosted/ir/soir_core.sio \
    self-hosted/ir/heap_storage.sio \
    self-hosted/ir/mod.sio \
    self-hosted/ir/lower.sio \
    self-hosted/parser/ast.sio \
    self-hosted/parser/items.sio \
    self-hosted/compiler/main.sio \
    self-hosted/native/codegen_x86_linux.sio \
    self-hosted/native/runtime_context.sio \
    scripts/ci/madaros_global_init_transport_gate.sh \
    scripts/bootstrap/bootstrap_concat.sh; then
  fail legacy_or_default_surface_changed
fi

SOIR_WRITER_STATIC_ONLY=1 SOUC="$SOUC" bash scripts/ci/soir_writer_v0_gate.sh >"$TMP/legacy-writer-static.log" 2>&1 || {
  cat "$TMP/legacy-writer-static.log" >&2
  fail legacy_writer_static_contract
}

for source in "$ARENA" "$WRITER" "$BRIDGE"; do
  "$SOUC" check "$source" >"$TMP/$(basename "$source").check.log" 2>&1 || {
    cat "$TMP/$(basename "$source").check.log" >&2
    fail "source_check_${source//\//_}"
  }
done

compose_and_run() {
  local witness="$1"
  local marker="$2"
  local name composite elf log
  name="$(basename "$witness" .sio)"
  composite="$(mktemp "$ROOT/self-hosted/ir/${name}.XXXXXX.sio")"
  COMPOSITES+=("$composite")
  elf="$TMP/$name.elf"
  log="$TMP/$name.log"

  {
    printf 'module ir::%s_gate\n\n' "$name"
    grep -E '^pub let SOIR_(MAX_SIZE|EMPTY_EXTENSION_COUNT_FIELDS):' "$CORE"
    awk '/SOIR_WRITER_PLAN_CORE_BEGIN/{p=1} p{print} /SOIR_WRITER_PLAN_CORE_END/{p=0}' "$WRITER"
    awk '/SOIR_WRITER_SHARED_SCALAR_PRIMITIVES_BEGIN/{p=1} p{print} /SOIR_WRITER_SHARED_SCALAR_PRIMITIVES_END/{p=0}' "$WRITER"
    awk '/SOIR_WRITER_SCALAR_EMPTY_CORE_BEGIN/{p=1} p{print} /SOIR_WRITER_SCALAR_EMPTY_CORE_END/{p=0}' "$WRITER"
    sed '/^module ir::arena_v2_shadow$/d' "$ARENA"
    sed -e '/^module ir::arena_v2_soir_bridge$/d' \
        -e '/^use ir::arena_v2_shadow::\*$/d' \
        -e '/^use ir::soir_writer::\*$/d' "$BRIDGE"
    sed -e '/^use ir::arena_v2_shadow::\*$/d' \
        -e '/^use ir::soir_writer::\*$/d' \
        -e '/^use ir::arena_v2_soir_bridge::\*$/d' "$witness"
  } >"$composite"

  "$SOUC" check "$composite" >"$log" 2>&1 || {
    cat "$log" >&2
    fail "composite_check_$name"
  }
  "$SOUC" --native-v2-compile "$composite" -o "$elf" >>"$log" 2>&1 || {
    cat "$log" >&2
    fail "composite_build_$name"
  }
  chmod +x "$elf"
  "$elf" >"$TMP/$name.out" 2>&1 || {
    cat "$TMP/$name.out" >&2
    fail "composite_runtime_$name"
  }
  grep -Fxq "$marker" "$TMP/$name.out" || {
    cat "$TMP/$name.out" >&2
    fail "marker_missing_$name"
  }
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

compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_CHECK source=pass composite_matrix=pass legacy_writer_static=pass'
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_PLAN storage=private_scalar_columns capacity=2 identity=slot,generation binding=arena,module_slot,module_generation,mutation_epoch,start,capacity,required,end,version release=generational output_publish=success_only aggregate_plan_cross_call=none'
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_EMIT deterministic=pass capacity_below=reject capacity_exact=pass no_partial_write=pass origin=zero_only'
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_IDS invalid=reject stale=reject reuse=reject cross_arena=reject mutation_after_preflight=reject reversible_mutation=reject sequential_stale_then_reuse=pass mutation_repreflight=reject'
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_PLAN_LIFECYCLE capacity_exhaustion=distinct failed_same_out=preserved release=pass generation_reuse=pass'
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_WIRE owner=soir_writer scalar_core=mechanically_extracted core_constants=max_size,extension_count empty_size_formula=writer_local bytes=self_repeat_only'
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_DIFFERENTIAL mode=shadow_canonical_not_differential meaning=writer_owned_path_not_byte_parity legacy_runtime=failed legacy_final_fn_count=2048 legacy_stack_frames=404144224'
printf '%s\n' 'IR_MODULE_ARENA_V2_SOIR_V5_BLOCKER nonzero_start=nested_mutable_buffer_forwarding legacy_oracle=large_irmodule_transport full_imported_closure=backend_rc19'
printf 'IR_MODULE_ARENA_V2_SOIR_V5_PASS compiler=%s compiler_sha256=%s base=%s\n' "$SOUC" "$compiler_sha256" "$BASE"
