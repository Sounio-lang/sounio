#!/usr/bin/env bash
# scripts/ci/ontology_frontiers_gate.sh
#
# CI gate for the ontology-frontier research prototypes
# (artifacts/ontology-frontiers/). For each prototype .sio this gate:
#   (a) runs `./bin/souc check <file>` and requires `check: OK` in the output
#       (and no `parse error`),
#   (b) runs `./bin/souc run <file>` and requires an exact `ALL PASS` output
#       line.
#
# souc's process exit code is unreliable for both subcommands, so all
# verdicts are derived from captured stdout/stderr, never from $?.
# A per-file OK/FAIL line is printed; the gate exits 1 if any file fails.
#
# Runs from any cwd (all paths anchored at ROOT_DIR). Each `souc run` takes
# ~30-60s; a per-file timeout guards against hangs.
#
# Env overrides:
#   SOUC_BIN                          compiler wrapper (default: bin/souc)
#   ONTOLOGY_FRONTIERS_RUN_TIMEOUT    per-file `souc run` timeout, seconds
#                                     (default: 300)
#
# Exit 0 = all prototypes pass. Exit 1 = at least one prototype failed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
RUN_TIMEOUT="${ONTOLOGY_FRONTIERS_RUN_TIMEOUT:-300}"

PROTOTYPES=(
    "artifacts/ontology-frontiers/epistemic-alignment-repair/alignment_repair.sio"
    "artifacts/ontology-frontiers/epistemic-claim-status/claim_status.sio"
    "artifacts/ontology-frontiers/epistemic-claim-status/interval_claims.sio"
    "artifacts/ontology-frontiers/consistent-ontology-evolution/version_chain.sio"
    "artifacts/ontology-frontiers/consistent-ontology-evolution/version_chain_removal.sio"
    "artifacts/ontology-frontiers/consistent-ontology-evolution/minimal_repair_demo.sio"
    "artifacts/ontology-frontiers/el-grounding/el_conflict_demo.sio"
    "artifacts/ontology-frontiers/epistemic-alignment-repair/tie_repair_demo.sio"
    "artifacts/ontology-frontiers/real-data/real_repair_driver.sio"
    "artifacts/ontology-frontiers/real-data/scale/full_scale_driver.sio"
    "artifacts/ontology-frontiers/real-data/scale/elplus_scale_driver.sio"
    "artifacts/ontology-frontiers/real-data/scale/go_elplus_driver.sio"
    "artifacts/ontology-frontiers/real-data/scale/go_full_elplus_driver.sio"
    "examples/ontology_pipeline_demo.sio"
)

TMP_DIR="$(mktemp -d /tmp/ontology-frontiers-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0

gate_file() {
    local rel="$1"
    local src="$ROOT_DIR/$rel"
    local name
    name="$(basename "$rel" .sio)"
    local problems=0

    if [[ ! -f "$src" ]]; then
        printf '[ontology-frontiers] FAIL: %s (missing file: %s)\n' "$name" "$rel" >&2
        FAIL=$((FAIL + 1))
        return
    fi

    # (a) souc check — exit code unreliable, grep stdout.
    local check_log="$TMP_DIR/$name.check.log"
    "$SOUC_BIN" check "$src" >"$check_log" 2>&1 || true
    if grep -Fq 'parse error' "$check_log"; then
        printf '[ontology-frontiers]   check: parse error (see %s)\n' "$check_log" >&2
        problems=$((problems + 1))
    elif ! grep -Fq 'check: OK' "$check_log"; then
        printf '[ontology-frontiers]   check: missing "check: OK" (see %s)\n' "$check_log" >&2
        problems=$((problems + 1))
    fi

    # (b) souc run — exit code unreliable, require an exact ALL PASS line.
    local run_log="$TMP_DIR/$name.run.log"
    timeout "$RUN_TIMEOUT" "$SOUC_BIN" run "$src" >"$run_log" 2>&1 || true
    if ! grep -Fxq 'ALL PASS' "$run_log"; then
        printf '[ontology-frontiers]   run: missing exact "ALL PASS" line (see %s)\n' "$run_log" >&2
        problems=$((problems + 1))
    fi

    if [[ "$problems" -eq 0 ]]; then
        PASS=$((PASS + 1))
        printf '[ontology-frontiers] OK:   %s\n' "$rel"
    else
        FAIL=$((FAIL + 1))
        printf '[ontology-frontiers] FAIL: %s (%d problem(s))\n' "$rel" "$problems" >&2
    fi
}

echo "[ontology-frontiers] souc=$SOUC_BIN"
echo "[ontology-frontiers] prototypes=${#PROTOTYPES[@]} run_timeout=${RUN_TIMEOUT}s"

if [[ ! -x "$SOUC_BIN" ]]; then
    echo "[ontology-frontiers] ERROR: souc wrapper not executable: $SOUC_BIN" >&2
    exit 1
fi

for rel in "${PROTOTYPES[@]}"; do
    gate_file "$rel"
done

printf '[ontology-frontiers] results: %d passed, %d failed\n' "$PASS" "$FAIL"

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
