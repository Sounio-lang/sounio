#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

native_out="$(mktemp)"
legacy_out="$(mktemp)"
trap 'rm -f "$native_out" "$legacy_out"' EXIT

./bin/souc run examples/hello.sio >"$native_out"
grep -q "Hello, Sounio" "$native_out"

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/epistemic_kan.sio >"$legacy_out"
grep -q "ALL PASS" "$legacy_out"

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/epistemic_kan_fixed_point.sio >"$legacy_out"
grep -q "ALL PASS" "$legacy_out"

./scripts/ci/ekan_native_frontier_matrix.sh >"$native_out"
grep -q "EKAN_NATIVE_FRONTIER_MATRIX_PASS" "$native_out"

./scripts/ci/ekan_native_blocker_probe.sh >"$native_out"
grep -q "EKAN_NATIVE_BLOCKER_PROBE_PASS" "$native_out"

./scripts/ci/ekan_native_readout_probe.sh >"$native_out"
grep -q "EKAN_NATIVE_READOUT_PROBE_PASS" "$native_out"

./scripts/ci/ekan_native_width_probe.sh >"$native_out"
grep -q "EKAN_NATIVE_WIDTH_PROBE_PASS" "$native_out"

./scripts/ci/ekan_native_input_width_probe.sh >"$native_out"
grep -q "EKAN_NATIVE_INPUT_WIDTH_PROBE_PASS" "$native_out"

echo "EKAN_NATIVE_CORE_GATE_PASS"
