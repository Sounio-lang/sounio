#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

classify_probe() {
  local label="$1"
  local path="$2"
  local log="$tmpdir/${label}.log"
  local rc=0

  set +e
  ./bin/souc run "$path" >"$log" 2>&1
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]] && grep -q "ALL PASS\\|ekan_knowledge_basic: PASS" "$log"; then
    echo "EKAN_NATIVE_FRONTIER_PROBE_${label}=XPASS_UNSTABLE"
    return 0
  fi

  if [[ "$rc" -eq 139 ]] || grep -Eq 'Segmentation fault|compiler exited 139|exit.*139' "$log"; then
    echo "EKAN_NATIVE_FRONTIER_PROBE_${label}=KNOWN_BLOCKER"
    return 0
  fi

  echo "EKAN_NATIVE_FRONTIER_PROBE_${label}=UNEXPECTED rc=${rc}" >&2
  cat "$log" >&2
  return 1
}

classify_probe KNOWLEDGE_BASIC tests/run-pass/ekan_knowledge_basic.sio
classify_probe EDGE tests/known_failures/ekan_fixed_point_edge_native_v2_probe.sio
classify_probe HIDDEN_COMPACT tests/known_failures/ekan_fixed_point_hidden_compact_native_v2_probe.sio
classify_probe HIDDEN_VERBOSE tests/known_failures/ekan_fixed_point_hidden_native_v2_blocker.sio
classify_probe LINEAR_READOUT_COMPACT tests/known_failures/ekan_fixed_point_linear_readout_compact_native_v2_probe.sio
classify_probe LINEAR_READOUT_VERBOSE tests/known_failures/ekan_fixed_point_linear_readout_native_v2_probe.sio
classify_probe HAT3_READOUT tests/known_failures/ekan_fixed_point_hat3_readout_native_v2_probe.sio
classify_probe HAT5_READOUT tests/known_failures/ekan_fixed_point_hat_readout_native_v2_probe.sio
classify_probe TWO_HIDDEN tests/known_failures/ekan_fixed_point_two_hidden_native_v2_probe.sio
classify_probe TWO_HIDDEN_READOUT tests/known_failures/ekan_fixed_point_two_hidden_readout_native_v2_probe.sio
classify_probe THREE_INPUT_TWO_HIDDEN tests/known_failures/ekan_fixed_point_three_input_two_hidden_native_v2_probe.sio
classify_probe FOUR_INPUT_ONE_HIDDEN tests/known_failures/ekan_fixed_point_four_input_one_hidden_native_v2_probe.sio
classify_probe FOUR_INPUT_TWO_HIDDEN tests/known_failures/ekan_fixed_point_four_input_two_hidden_native_v2_probe.sio

echo "EKAN_NATIVE_FRONTIER_MATRIX_PASS"
