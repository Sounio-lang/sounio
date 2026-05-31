#!/usr/bin/env bash
# epistemic_tensor_dyn_gate.sh — gate for the dynamic-width epistemic tensor
# ("the dragon") and the typed dose-certification boundary.
#
# Proves, end to end, on the canonical compiler:
#   1. the heap epistemic matmul math is correct                 (epi_matmul_probe)
#   2. the module (matmul + GUM unc + conf + bias + sigmoid)     (epi_tensor_dyn_basic)
#   3. a width-512 layer runs — impossible under the old 256 cap (epi_capacity_512)  ← dragon dead
#   4. the typed dose contract ACCEPTS a literal-ε certification (dose_certify_accept)
#   5. the typed dose contract REJECTS a computed/learned ε      (dose_contract_bypass) ← typed refusal
#
# Usage: scripts/ci/epistemic_tensor_dyn_gate.sh [path-to-souc]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${1:-$ROOT/bin/souc}"
TMP="$(mktemp -d)"
fails=0

if [[ ! -x "$SOUC" ]]; then echo "FAIL: souc not found/executable at $SOUC"; exit 1; fi
echo "souc: $SOUC ($(md5sum "$SOUC" | cut -d' ' -f1))"

run_pass() { # $1 src, $2 expected-token
  local src="$ROOT/$1" tok="$2" elf="$TMP/$(basename "$1").elf"
  if ! "$SOUC" "$src" "$elf" >"$TMP/c.log" 2>&1; then
    echo "  FAIL(compile) $1"; grep -iE 'error' "$TMP/c.log" | head -2; fails=$((fails+1)); return
  fi
  chmod +x "$elf"
  local out; out="$("$elf" 2>&1)"
  if echo "$out" | grep -q "$tok"; then echo "  PASS $1"; else echo "  FAIL(run) $1 — got: $out"; fails=$((fails+1)); fi
}

compile_fail() { # $1 src, $2 expected error pattern
  local src="$ROOT/$1" pat="$2" elf="$TMP/$(basename "$1").elf"
  rm -f "$elf"
  if "$SOUC" "$src" "$elf" >"$TMP/c.log" 2>&1 || [[ -f "$elf" ]]; then
    echo "  FAIL $1 — compiled clean (contract bypassable!)"; fails=$((fails+1)); return
  fi
  if grep -qi "$pat" "$TMP/c.log"; then echo "  PASS $1 (rejected: $pat)"; else
    echo "  FAIL $1 — rejected but not by '$pat'"; tail -3 "$TMP/c.log" | sed 's/^/    /'; fails=$((fails+1)); fi
}

echo "── the dragon ──"
run_pass tests/run-pass/epi_matmul_probe.sio          EPI_MATMUL_PROBE_PASS
run_pass tests/run-pass/epi_tensor_dyn_basic.sio      EPI_TENSOR_DYN_BASIC_PASS
run_pass tests/run-pass/epi_tensor_capacity_gt256.sio EPI_CAPACITY_512_PASS
echo "── the typed refusal ──"
run_pass     tests/run-pass/dose_certify_accept.sio     DOSE_CERTIFY_ACCEPT_PASS
compile_fail tests/compile-fail/dose_contract_bypass.sio "EpistemicComplete violation"
echo "── the wiring (dragon's propagated confidence drives the refusal) ──"
run_pass     tests/run-pass/dose_from_epinet.sio        DOSE_FROM_EPINET_PASS
echo "── enablement: a TRAINED epistemic net on the substrate ──"
run_pass     tests/run-pass/epinet_train_vanco.sio      EPINET_TRAIN_VANCO_PASS
echo "── SOTA: refinement-typed graph, type-bound ≥ runtime uncertainty (sound) ──"
run_pass     tests/run-pass/typed_epi_graph_soundness.sio  TYPED_EPI_GRAPH_SOUNDNESS_PASS

rm -rf "$TMP"
echo ""
if [[ $fails -ne 0 ]]; then echo "epistemic_tensor_dyn_gate: FAIL ($fails)"; exit 1; fi
echo "epistemic_tensor_dyn_gate: PASS (8/8)"
