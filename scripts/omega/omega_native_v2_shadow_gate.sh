#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
NATIVE_V2_TARGET="${SOUNIO_NATIVE_V2_TARGET:-x86_64-linux}"
OUT_DIR="$ROOT_DIR/artifacts/omega"
SELFTEST_LOG="$(mktemp)"
SMOKE_LOG="$(mktemp)"
GC_RETRY_LOG="$(mktemp)"
trap 'rm -f "$SELFTEST_LOG" "$SMOKE_LOG" "$GC_RETRY_LOG"' EXIT

mkdir -p "$OUT_DIR"

SOUNIO_NATIVE_V2_TARGET=x86_64-linux "$SOUC_BIN" run "$ROOT_DIR/self-hosted/compiler/main.sio" -- --self-test >"$SELFTEST_LOG" 2>&1

grep -q 'compiler/main tests: 225/225 passed' "$SELFTEST_LOG"

if [[ "${SOUNIO_RUN_NATIVE_V2_PREVIEW_SMOKE:-0}" == "1" || "$NATIVE_V2_TARGET" != "x86_64-linux" ]]; then
  SOUNIO_NATIVE_V2_TARGET="$NATIVE_V2_TARGET" "$SOUC_BIN" run "$ROOT_DIR/self-hosted/compiler/main.sio" -- --native-v2-preview-smoke >"$SMOKE_LOG" 2>&1
  grep -q 'native_v2_preview_smoke: ok' "$SMOKE_LOG"
fi

if [[ "$NATIVE_V2_TARGET" == "x86_64-linux" ]]; then
  "$SOUC_BIN" run "$ROOT_DIR/self-hosted/compiler/main.sio" -- --native-v2-gc-retry-smoke >"$GC_RETRY_LOG" 2>&1
  grep -q 'native_v2_gc_retry_smoke: ok' "$GC_RETRY_LOG"
fi

CONTRACT_SELFTEST="$OUT_DIR/native_backend_v2_contract.selftest.json"
SHADOW_CONTRACT_SELFTEST="$OUT_DIR/native_backend_v2_shadow_contract.selftest.json"
if [[ "$NATIVE_V2_TARGET" == "x86_64-linux" ]]; then
  CONTRACT_TARGET="$CONTRACT_SELFTEST"
  CONTRACT_V1="$OUT_DIR/native_backend_v2_contract.v1.json"
  SMOKE_BIN="$OUT_DIR/native_backend_v2_scalar_smoke.selftest.bin"
else
  CONTRACT_TARGET="$OUT_DIR/native_backend_v2_contract.$NATIVE_V2_TARGET.json"
  CONTRACT_V1="$CONTRACT_TARGET"
  SMOKE_BIN="$OUT_DIR/native_backend_v2_scalar_smoke.$NATIVE_V2_TARGET.bin"
fi
SHADOW_CONTRACT_V1="$OUT_DIR/native_backend_v2_shadow_contract.v1.json"
GC_RETRY_SMOKE_BIN="$OUT_DIR/native_backend_v2_gc_retry_smoke.selftest.bin"

test -s "$CONTRACT_SELFTEST"
test -s "$SHADOW_CONTRACT_SELFTEST"
test -s "$CONTRACT_TARGET"
test -s "$SMOKE_BIN"
if [[ "$NATIVE_V2_TARGET" == "x86_64-linux" ]]; then
  test -s "$GC_RETRY_SMOKE_BIN"
fi

grep -q '"coverage":"scalar_core"' "$CONTRACT_TARGET"
grep -q '"fail_closed":true' "$CONTRACT_TARGET"
grep -q '"legacy_fallback":false' "$CONTRACT_TARGET"
grep -q '"stack_map_records":1' "$CONTRACT_TARGET"
grep -q '"safepoint_count":1' "$CONTRACT_TARGET"
grep -q '"root_map_schema":"v2"' "$CONTRACT_TARGET"
grep -q '"managed_handles":true' "$CONTRACT_TARGET"
grep -q '"descriptor_tables":true' "$CONTRACT_TARGET"
grep -q '"gc_mark_compact_model":true' "$CONTRACT_TARGET"
grep -q '"gc_precise_descriptor_scanning":true' "$CONTRACT_TARGET"
grep -q '"gc_handle_relocation_model":true' "$CONTRACT_TARGET"
grep -q '"ffi_pinning_model":true' "$CONTRACT_TARGET"
grep -q "\"resolved_target\":\"$NATIVE_V2_TARGET\"" "$CONTRACT_TARGET"

if [[ "$NATIVE_V2_TARGET" == "x86_64-linux" ]]; then
  grep -q '"runtime_metadata_embedded":true' "$CONTRACT_TARGET"
  grep -q '"gc_state_embedded":true' "$CONTRACT_TARGET"
  grep -q '"gc_slow_path_active":true' "$CONTRACT_TARGET"
  grep -q '"stack_maps_runtime_active":true' "$CONTRACT_TARGET"
  grep -q '"root_map_fields_embedded":true' "$CONTRACT_TARGET"
  grep -q '"coarse_root_kind_tags":true' "$CONTRACT_TARGET"
  grep -q '"deopt_ids_embedded":true' "$CONTRACT_TARGET"
  grep -q '"osr_eligibility_embedded":true' "$CONTRACT_TARGET"
  grep -q '"handle_indirection":true' "$CONTRACT_TARGET"
  grep -q '"descriptor_tables_embedded":true' "$CONTRACT_TARGET"
  grep -q '"gc_runtime_retry_active":true' "$CONTRACT_TARGET"
  grep -q '"gc_current_frame_root_scan":true' "$CONTRACT_TARGET"
else
  grep -q '"runtime_metadata_embedded":false' "$CONTRACT_TARGET"
  grep -q '"gc_state_embedded":false' "$CONTRACT_TARGET"
  grep -q '"stack_maps_runtime_active":false' "$CONTRACT_TARGET"
fi

GC_RETRY_STATUS=0
if [[ "$NATIVE_V2_TARGET" == "x86_64-linux" ]]; then
  set +e
  "$GC_RETRY_SMOKE_BIN"
  GC_RETRY_STATUS=$?
  set -e
  [[ "$GC_RETRY_STATUS" -eq 23 ]]
fi

if [[ "$NATIVE_V2_TARGET" == "x86_64-linux" ]]; then
  cp "$CONTRACT_SELFTEST" "$CONTRACT_V1"
fi
cp "$SHADOW_CONTRACT_SELFTEST" "$SHADOW_CONTRACT_V1"

if [[ "$NATIVE_V2_TARGET" == "x86_64-linux" ]]; then
  GATE_V1="$OUT_DIR/native_backend_v2_gate.v1.json"
else
  GATE_V1="$OUT_DIR/native_backend_v2_gate.$NATIVE_V2_TARGET.json"
fi

cat >"$GATE_V1" <<EOF
{
  "schema": "sounio.native_backend_v2_gate.v1",
  "backend": "native-v2",
  "target": "$NATIVE_V2_TARGET",
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
  "gc_runtime_retry_active": $([[ "$NATIVE_V2_TARGET" == "x86_64-linux" ]] && echo true || echo false),
  "gc_current_frame_root_scan": $([[ "$NATIVE_V2_TARGET" == "x86_64-linux" ]] && echo true || echo false),
  "gc_retry_smoke_exit_code": $GC_RETRY_STATUS
}
EOF

# Sprint 87: shadow alias retired — native-v2 now handles float ops directly.
# The shadow contract previously mirrored the main gate; it is no longer needed
# since --native-compile no longer falls back to legacy for float-bearing .sio files.

printf '[native-v2-gate] ok target=%s\n' "$NATIVE_V2_TARGET"
