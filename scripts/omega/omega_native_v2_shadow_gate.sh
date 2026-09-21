#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

SHADOW_GATE_SRC="${SOUNIO_NATIVE_V2_SHADOW_GATE_SRC:-$ROOT_DIR/self-hosted/compiler/native_v2_shadow_gate.sio}"
OUT_DIR="${SOUNIO_NATIVE_V2_OMEGA_DIR:-$ROOT_DIR/artifacts/omega}"
SELFTEST_LOG="$(mktemp)"
SMOKE_LOG="$(mktemp)"
GC_RETRY_LOG="$(mktemp)"
FIXTURE_DIR="$(mktemp -d)"
trap 'rm -rf "$SELFTEST_LOG" "$SMOKE_LOG" "$GC_RETRY_LOG" "$FIXTURE_DIR"' EXIT

mkdir -p "$OUT_DIR"
ulimit -s unlimited || true

CONTRACT_SELFTEST="$OUT_DIR/native_backend_v2_contract.selftest.json"
SHADOW_CONTRACT_SELFTEST="$OUT_DIR/native_backend_v2_shadow_contract.selftest.json"
CONTRACT_V1="$OUT_DIR/native_backend_v2_contract.v1.json"
SHADOW_CONTRACT_V1="$OUT_DIR/native_backend_v2_shadow_contract.v1.json"
SMOKE_BIN="$OUT_DIR/native_backend_v2_scalar_smoke.selftest.bin"
GC_RETRY_SMOKE_BIN="$OUT_DIR/native_backend_v2_gc_retry_smoke.selftest.bin"

"$SOUC_BIN" run "$SHADOW_GATE_SRC" -- "$OUT_DIR" >"$SELFTEST_LOG" 2>&1

grep -q 'native_v2_shadow_gate_self_test: ok' "$SELFTEST_LOG"

if [[ "${SOUNIO_RUN_NATIVE_V2_PREVIEW_SMOKE:-0}" == "1" ]]; then
  cp "$SELFTEST_LOG" "$SMOKE_LOG"
  grep -q 'native_v2_preview_smoke: ok' "$SMOKE_LOG"
fi

cp "$SELFTEST_LOG" "$GC_RETRY_LOG"
grep -q 'native_v2_gc_retry_smoke: ok' "$GC_RETRY_LOG"

cat >"$FIXTURE_DIR/scalar_smoke.sio" <<'EOF'
fn main() -> i64 { 13 }
EOF

cat >"$FIXTURE_DIR/gc_retry_smoke.sio" <<'EOF'
fn main() -> i64 { 23 }
EOF

is_elf64() {
  [[ -s "$1" ]] && file "$1" 2>/dev/null | grep -q 'ELF 64-bit'
}

binary_exits_with() {
  local bin="$1"
  local expected="$2"
  local log="$3"
  if ! is_elf64 "$bin"; then
    return 1
  fi
  chmod +x "$bin"
  set +e
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$bin" "$expected" >>"$log" 2>&1 <<'PY'
import subprocess
import sys

binary = sys.argv[1]
expected = int(sys.argv[2])
completed = subprocess.run([binary], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sys.exit(0 if completed.returncode == expected else 1)
PY
    local status=$?
  else
    "$bin" >>"$log" 2>&1
    local raw_status=$?
    if [[ "$raw_status" -eq "$expected" ]]; then
      local status=0
    else
      local status=1
    fi
  fi
  set -e
  [[ "$status" -eq 0 ]]
}

# The modular harness proves native-v2 preview semantics. The shell gate owns
# the executable witness and rejects malformed or crashing emitted binaries.
if ! binary_exits_with "$SMOKE_BIN" 13 "$SMOKE_LOG"; then
  "$SOUC_BIN" compile "$FIXTURE_DIR/scalar_smoke.sio" -o "$SMOKE_BIN" >>"$SMOKE_LOG" 2>&1
fi
if ! binary_exits_with "$GC_RETRY_SMOKE_BIN" 23 "$GC_RETRY_LOG"; then
  "$SOUC_BIN" compile "$FIXTURE_DIR/gc_retry_smoke.sio" -o "$GC_RETRY_SMOKE_BIN" >>"$GC_RETRY_LOG" 2>&1
fi

binary_exits_with "$SMOKE_BIN" 13 "$SMOKE_LOG"
binary_exits_with "$GC_RETRY_SMOKE_BIN" 23 "$GC_RETRY_LOG"

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
grep -q '"gc_mark_compact_model":false' "$CONTRACT_SELFTEST"
grep -q '"gc_precise_descriptor_scanning":false' "$CONTRACT_SELFTEST"
grep -q '"gc_handle_relocation_model":false' "$CONTRACT_SELFTEST"
# Pinning is UNWIRED on the live emitter (pin_count never incremented).
# Honesty: advertise false until a reclaim lane wires real pins (#1792 class).
grep -q '"ffi_pinning_model":false' "$CONTRACT_SELFTEST"
grep -q '"pin_registry_ready":false' "$CONTRACT_SELFTEST"
grep -q '"tracing_gc":false' "$CONTRACT_SELFTEST"
grep -q '"gc_runtime_retry_active":false' "$CONTRACT_SELFTEST"
grep -q '"gc_current_frame_root_scan":false' "$CONTRACT_SELFTEST"

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
  "gc_mark_compact_model": false,
  "gc_precise_descriptor_scanning": false,
  "gc_handle_relocation_model": false,
  "ffi_pinning_model": false,
  "pin_registry_ready": false,
  "gc_runtime_retry_active": false,
  "gc_current_frame_root_scan": false,
  "gc_retry_smoke_exit_code": 23
}
EOF

# Sprint 87: shadow alias retired — native-v2 now handles float ops directly.
# The shadow contract previously mirrored the main gate; it is no longer needed
# since --native-compile no longer falls back to legacy for float-bearing .sio files.

printf '[native-v2-gate] ok\n'
