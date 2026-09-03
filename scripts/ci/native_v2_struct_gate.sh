#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_STRUCT_GATE_DIR:-$(mktemp -d /tmp/sounio-native-v2-struct.XXXXXX)}"
BIN="$OUT_DIR/struct_basic"
CHECK_LOG="$OUT_DIR/driver.check.log"
COMPILE_LOG="$OUT_DIR/struct_basic.compile.log"
STDOUT_LOG="$OUT_DIR/struct_basic.stdout"
EXPECTED_LOG="$OUT_DIR/struct_basic.expected"

mkdir -p "$OUT_DIR"

run_sub_gate() {
  local name="$1"
  shift
  echo "[native-v2-struct] running $name"
  "$@"
}

printf '[native-v2-struct] souc=%s\n' "$SOUC_BIN"
printf '[native-v2-struct] out=%s\n' "$OUT_DIR"

"$SOUC_BIN" check self-hosted/compiler/native_compile_driver.sio >"$CHECK_LOG" 2>&1

"$SOUC_BIN" run self-hosted/compiler/native_compile_driver.sio -- \
  examples/native/struct_basic.sio -o "$BIN" >"$COMPILE_LOG" 2>&1

if [[ ! -x "$BIN" ]]; then
  echo "[native-v2-struct] FAIL: generated binary not executable: $BIN" >&2
  tail -n 40 "$COMPILE_LOG" >&2 || true
  exit 1
fi

"$BIN" >"$STDOUT_LOG" 2>/dev/null
printf '7\n' >"$EXPECTED_LOG"

if ! cmp -s "$EXPECTED_LOG" "$STDOUT_LOG"; then
  echo "[native-v2-struct] FAIL: output mismatch" >&2
  echo "[native-v2-struct] expected: $(cat "$EXPECTED_LOG")" >&2
  echo "[native-v2-struct] got:      $(cat "$STDOUT_LOG")" >&2
  exit 1
fi

run_sub_gate serious_track bash scripts/ci/native_v2_serious_track_gate.sh

STRUCT_SUB_GATES=(
  native_v2_algebra_law_gate.sh
  native_v2_array_gate.sh
  native_v2_logical_gate.sh
  native_v2_enum_match_gate.sh
  native_v2_nested_field_gate.sh
  native_v2_struct_mutation_gate.sh
  native_v2_struct_param_gate.sh
  native_v2_struct_return_gate.sh
  native_v2_out_param_boundary_gate.sh
)

for gate in "${STRUCT_SUB_GATES[@]}"; do
  run_sub_gate "$gate" bash "scripts/ci/$gate"
done

# Avoid re-entering the CPU umbrella when struct_gate is already running under it.
run_sub_gate imported_core_abi env SOUNIO_NATIVE_V2_FRONTEND_RUN_CPU_UMBRELLA=0 \
  bash scripts/ci/native_v2_imported_core_abi_gate.sh
run_sub_gate imported_hof_abi env SOUNIO_NATIVE_V2_FRONTEND_RUN_CPU_UMBRELLA=0 \
  bash scripts/ci/native_v2_imported_hof_abi_gate.sh

run_sub_gate metal_algebra bash scripts/ci/native_v2_metal_algebra_gate.sh

if [[ "${SOUNIO_NATIVE_V2_NVIDIA_BARE_METAL_GATE_RUN:-0}" == "1" ]]; then
  run_sub_gate nvidia_bare_metal bash scripts/ci/native_v2_nvidia_bare_metal_gate.sh
else
  echo "[native-v2-struct] skipping nvidia_bare_metal (set SOUNIO_NATIVE_V2_NVIDIA_BARE_METAL_GATE_RUN=1 on GPU hosts)"
fi

echo "[native-v2-struct] PASS: orchestrated struct/native-v2 regression suite"
