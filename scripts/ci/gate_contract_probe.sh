#!/usr/bin/env bash
# gate_contract_probe.sh — census of green-without-measure gate shapes.
#
# Observational by default (exit 0). Set GATE_CONTRACT_PROBE_FAIL=1 to exit 1
# when a WIRED gate matches a hard unmeasure class (U1 skip-green, U2 soft-or).
#
# Taxonomy: docs/audit/GATE_UNMEASURE_TAXONOMY_2026-08-17.md
# Complements: scripts/ci/gate_vacuity_gate.sh (U0 empty extraction only).
#
# Implementation note: bulk rg only — no per-file loops over 400+ gates.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FAIL_MODE="${GATE_CONTRACT_PROBE_FAIL:-0}"
OUT_DIR="${GATE_CONTRACT_PROBE_DIR:-$(mktemp -d /tmp/sounio-gate-contract-probe.XXXXXX)}"
mkdir -p "$OUT_DIR"

git ls-files 'scripts/ci/*_gate.sh' | sort -u >"$OUT_DIR/all_gates.txt"

{
  rg -o --no-filename 'scripts/ci/[A-Za-z0-9_./-]+' .github/workflows --glob '*.yml' 2>/dev/null || true
} | tr -d '"'"'"':' | rg '_gate\.sh$' | sort -u >"$OUT_DIR/wired_gates.txt" || true

comm -23 "$OUT_DIR/all_gates.txt" "$OUT_DIR/wired_gates.txt" >"$OUT_DIR/unwired_gates.txt" || true

# U1: files that both exit 0 and mention SKIP*
rg -l 'exit 0' -g '*_gate.sh' scripts/ci 2>/dev/null | sort -u >"$OUT_DIR/u1_exit0.txt" || true
rg -li 'SKIP|SKIPPED|skipping' -g '*_gate.sh' scripts/ci 2>/dev/null | sort -u >"$OUT_DIR/u1_skip_lang.txt" || true
comm -12 "$OUT_DIR/u1_exit0.txt" "$OUT_DIR/u1_skip_lang.txt" >"$OUT_DIR/u1_skip_exit0.txt" || true
comm -12 "$OUT_DIR/wired_gates.txt" "$OUT_DIR/u1_skip_exit0.txt" >"$OUT_DIR/wired_u1.txt" || true

# U2: soft-or in workflows
rg -n '\|\|[[:space:]]*echo' .github/workflows --glob '*.yml' 2>/dev/null \
  >"$OUT_DIR/u2_soft_or_workflows.txt" || true

# U3: mentions fixtures/tests paths, no missing-fail language, no -f test
rg -l 'FIXTURES=|fixtures/|/tests/' -g '*_gate.sh' scripts/ci 2>/dev/null | sort -u >"$OUT_DIR/u3_mentions_inputs.txt" || true
rg -l 'require_fixture|require_min_count|require_nonempty|fail ".*missing|FAIL:.*missing|gate_fail' -g '*_gate.sh' scripts/ci 2>/dev/null \
  | sort -u >"$OUT_DIR/u3_has_guard.txt" || true
rg -l '\[\[!? *-f |\[! -f |test -f |\[\[ -s ' -g '*_gate.sh' scripts/ci 2>/dev/null \
  | sort -u >"$OUT_DIR/u3_has_f_test.txt" || true
comm -23 "$OUT_DIR/u3_mentions_inputs.txt" "$OUT_DIR/u3_has_guard.txt" \
  | comm -23 - "$OUT_DIR/u3_has_f_test.txt" >"$OUT_DIR/u3_no_input_floor.txt" || true

# U7: skip Madaros/binary + OK/exit 0
rg -li 'skipping|binary missing|Madaros binary missing' -g '*_gate.sh' scripts/ci 2>/dev/null \
  | sort -u >"$OUT_DIR/u7_skip_lang.txt" || true
rg -l 'GATE_OK|exit 0' -g '*_gate.sh' scripts/ci 2>/dev/null | sort -u >"$OUT_DIR/u7_ok.txt" || true
comm -12 "$OUT_DIR/u7_skip_lang.txt" "$OUT_DIR/u7_ok.txt" >"$OUT_DIR/u7_partial_skip.txt" || true

# U9: KNOWN_BLOCKER
rg -l 'KNOWN_BLOCKER' -g '*_gate.sh' scripts/ci 2>/dev/null | sort -u >"$OUT_DIR/u9_known_blocker_green.txt" || true

# U10: lean_single forced without madaros dual path
rg -l 'SOUNIO_SOUC_ENGINE=lean_single' -g '*_gate.sh' scripts/ci 2>/dev/null | sort -u >"$OUT_DIR/u10_lean.txt" || true
rg -l 'default Madaros|GATE_ENGINE: both|SOUNIO_SOUC_ENGINE=madaros' -g '*_gate.sh' scripts/ci 2>/dev/null \
  | sort -u >"$OUT_DIR/u10_dual.txt" || true
comm -23 "$OUT_DIR/u10_lean.txt" "$OUT_DIR/u10_dual.txt" >"$OUT_DIR/u10_lean_single_only.txt" || true

# Contract headers
rg -l 'GATE_CONTRACT: v0' -g '*_gate.sh' scripts/ci 2>/dev/null | sort -u >"$OUT_DIR/has_contract_header.txt" || true
comm -23 "$OUT_DIR/wired_gates.txt" "$OUT_DIR/has_contract_header.txt" >"$OUT_DIR/wired_missing_contract.txt" || true

count() { wc -l <"$1" | tr -d ' '; }

n_all=$(count "$OUT_DIR/all_gates.txt")
n_wired=$(count "$OUT_DIR/wired_gates.txt")
n_unwired=$(count "$OUT_DIR/unwired_gates.txt")
n_u1=$(count "$OUT_DIR/u1_skip_exit0.txt")
n_wired_u1=$(count "$OUT_DIR/wired_u1.txt")
n_u2=$(count "$OUT_DIR/u2_soft_or_workflows.txt")
n_u3=$(count "$OUT_DIR/u3_no_input_floor.txt")
n_u7=$(count "$OUT_DIR/u7_partial_skip.txt")
n_u9=$(count "$OUT_DIR/u9_known_blocker_green.txt")
n_u10=$(count "$OUT_DIR/u10_lean_single_only.txt")
n_contract=$(count "$OUT_DIR/has_contract_header.txt")
n_wired_nc=$(count "$OUT_DIR/wired_missing_contract.txt")

echo "=== gate_contract_probe census ==="
echo "all_gates=$n_all"
echo "wired_paths_in_workflows=$n_wired"
echo "unwired_gates=$n_unwired"
echo "U1_skip_exit0=$n_u1 (wired∩U1=$n_wired_u1)"
echo "U2_soft_or_workflow_lines=$n_u2"
echo "U3_no_input_floor_heuristic=$n_u3"
echo "U7_partial_skip=$n_u7"
echo "U9_known_blocker=$n_u9"
echo "U10_lean_single_only=$n_u10"
echo "GATE_CONTRACT_v0_headers=$n_contract"
echo "wired_missing_contract_header=$n_wired_nc"
echo "out_dir=$OUT_DIR"
# U13: path-selected CI Decision (check-count fallacy documentation)
if [[ -f .github/workflows/ci.yml ]] && [[ -f scripts/ci/evaluate_ci_decision.py ]]; then
  if rg -q 'name: CI Decision|evaluate_ci_decision' .github/workflows/ci.yml \
    && rg -q 'selected' scripts/ci/evaluate_ci_decision.py; then
    echo "U13_defective_completeness_observer=present (Impact path-select + CI Decision selected-only)"
    echo "U13_note=raw check-run cardinality is NOT completeness; parse CI_DECISION_PASS selected="
  fi
fi

echo "taxonomy=docs/audit/GATE_UNMEASURE_TAXONOMY_2026-08-17.md"
echo "U0_empty_extraction=run scripts/ci/gate_vacuity_gate.sh separately"
echo "--- sample wired∩U1 (up to 8) ---"
head -8 "$OUT_DIR/wired_u1.txt" 2>/dev/null || true
echo "--- sample U2 lines (up to 8) ---"
head -8 "$OUT_DIR/u2_soft_or_workflows.txt" 2>/dev/null || true

hard=0
[[ "$n_wired_u1" -gt 0 ]] && hard=1
[[ "$n_u2" -gt 0 ]] && hard=1

if [[ "$FAIL_MODE" == "1" && "$hard" -eq 1 ]]; then
  echo "GATE_CONTRACT_PROBE_FAIL: wired unmeasure classes present (see $OUT_DIR)" >&2
  exit 1
fi

echo "GATE_CONTRACT_PROBE_OK mode=$([[ "$FAIL_MODE" == "1" ]] && echo enforce || echo observe)"
exit 0
