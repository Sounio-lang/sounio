#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-jit}"
OUT_DIR="$ROOT_DIR/artifacts/omega"
SELFTEST_LOG="$(mktemp)"
SMOKE_LOG="$(mktemp)"
GC_RETRY_LOG="$(mktemp)"
trap 'rm -f "$SELFTEST_LOG" "$SMOKE_LOG" "$GC_RETRY_LOG"' EXIT

mkdir -p "$OUT_DIR"

"$SOUC_BIN" run "$ROOT_DIR/self-hosted/compiler/main.sio" -- --self-test >"$SELFTEST_LOG" 2>&1

grep -q 'compiler/main tests: 225/225 passed' "$SELFTEST_LOG"

if [[ "${SOUNIO_RUN_NATIVE_V2_PREVIEW_SMOKE:-0}" == "1" ]]; then
  "$SOUC_BIN" run "$ROOT_DIR/self-hosted/compiler/main.sio" -- --native-v2-preview-smoke >"$SMOKE_LOG" 2>&1
  grep -q 'native_v2_preview_smoke: ok' "$SMOKE_LOG"
fi

"$SOUC_BIN" run "$ROOT_DIR/self-hosted/compiler/main.sio" -- --native-v2-gc-retry-smoke >"$GC_RETRY_LOG" 2>&1
grep -q 'native_v2_gc_retry_smoke: ok' "$GC_RETRY_LOG"

CONTRACT_SELFTEST="$OUT_DIR/native_backend_v2_contract.selftest.json"
SHADOW_CONTRACT_SELFTEST="$OUT_DIR/native_backend_v2_shadow_contract.selftest.json"
CONTRACT_V1="$OUT_DIR/native_backend_v2_contract.v1.json"
SHADOW_CONTRACT_V1="$OUT_DIR/native_backend_v2_shadow_contract.v1.json"
SMOKE_BIN="$OUT_DIR/native_backend_v2_scalar_smoke.selftest.bin"
GC_RETRY_SMOKE_BIN="$OUT_DIR/native_backend_v2_gc_retry_smoke.selftest.bin"

test -s "$CONTRACT_SELFTEST"
test -s "$SHADOW_CONTRACT_SELFTEST"
test -s "$SMOKE_BIN"
test -s "$GC_RETRY_SMOKE_BIN"

grep -q '"coverage":"scalar_core"' "$CONTRACT_SELFTEST"
grep -q '"fail_closed":true' "$CONTRACT_SELFTEST"
grep -q '"legacy_fallback":false' "$CONTRACT_SELFTEST"
grep -q '"stack_map_records":1' "$CONTRACT_SELFTEST"
grep -q '"safepoint_count":1' "$CONTRACT_SELFTEST"
grep -q '"runtime_metadata_embedded":true' "$CONTRACT_SELFTEST"
grep -q '"gc_state_embedded":true' "$CONTRACT_SELFTEST"
grep -q '"gc_slow_path_active":true' "$CONTRACT_SELFTEST"
grep -q '"stack_maps_runtime_active":true' "$CONTRACT_SELFTEST"
grep -q '"root_map_schema":"v2"' "$CONTRACT_SELFTEST"
grep -q '"root_map_fields_embedded":true' "$CONTRACT_SELFTEST"
grep -q '"coarse_root_kind_tags":true' "$CONTRACT_SELFTEST"
grep -q '"deopt_ids_embedded":true' "$CONTRACT_SELFTEST"
grep -q '"osr_eligibility_embedded":true' "$CONTRACT_SELFTEST"
grep -q '"managed_handles":true' "$CONTRACT_SELFTEST"
grep -q '"descriptor_tables":true' "$CONTRACT_SELFTEST"
grep -q '"handle_indirection":true' "$CONTRACT_SELFTEST"
grep -q '"descriptor_tables_embedded":true' "$CONTRACT_SELFTEST"
grep -q '"gc_mark_compact_model":true' "$CONTRACT_SELFTEST"
grep -q '"gc_precise_descriptor_scanning":true' "$CONTRACT_SELFTEST"
grep -q '"gc_handle_relocation_model":true' "$CONTRACT_SELFTEST"
grep -q '"ffi_pinning_model":true' "$CONTRACT_SELFTEST"
grep -q '"gc_runtime_retry_active":true' "$CONTRACT_SELFTEST"
grep -q '"gc_current_frame_root_scan":true' "$CONTRACT_SELFTEST"

set +e
"$GC_RETRY_SMOKE_BIN"
GC_RETRY_STATUS=$?
set -e
[[ "$GC_RETRY_STATUS" -eq 23 ]]

cp "$CONTRACT_SELFTEST" "$CONTRACT_V1"
cp "$SHADOW_CONTRACT_SELFTEST" "$SHADOW_CONTRACT_V1"

cat >"$OUT_DIR/native_backend_v2_gate.v1.json" <<'EOF'
{
  "schema": "sounio.native_backend_v2_gate.v1",
  "backend": "native-v2",
  "selftest_passed": true,
  "scalar_smoke_present": true,
  "machine_ir_coverage": "scalar_core",
  "fail_closed": true,
  "stack_map_records": 1,
  "safepoint_count": 1,
  "runtime_metadata_embedded": true,
  "gc_state_embedded": true,
  "gc_slow_path_active": true,
  "stack_maps_runtime_active": true,
  "root_map_schema": "v2",
  "root_map_fields_embedded": true,
  "coarse_root_kind_tags": true,
  "deopt_ids_embedded": true,
  "osr_eligibility_embedded": true,
  "managed_handles": true,
  "descriptor_tables_embedded": true,
  "handle_indirection": true,
  "gc_mark_compact_model": true,
  "gc_precise_descriptor_scanning": true,
  "gc_handle_relocation_model": true,
  "ffi_pinning_model": true,
  "gc_runtime_retry_active": true,
  "gc_current_frame_root_scan": true,
  "gc_retry_smoke_exit_code": 23
}
EOF

# Sprint 87: shadow alias retired — native-v2 now handles float ops directly.
# The shadow contract previously mirrored the main gate; it is no longer needed
# since --native-compile no longer falls back to legacy for float-bearing .sio files.

printf '[native-v2-gate] ok\n'
