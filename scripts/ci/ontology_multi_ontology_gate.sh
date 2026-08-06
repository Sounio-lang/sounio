#!/usr/bin/env bash
# scripts/ci/ontology_multi_ontology_gate.sh
#
# CI gate for the round-13 multi-ontology EL+ role-aware closure drivers
# (artifacts/ontology-frontiers/multi-ontology/). Same protocol as
# scripts/ci/ontology_frontiers_gate.sh: for each driver this gate
#   (a) runs `./bin/souc check <file>` and requires `check: OK` in the output
#       (and no `parse error`),
#   (b) runs `./bin/souc run <file>` and requires an exact `ALL PASS`
#       output line.
#
# Kept SEPARATE from ontology_frontiers_gate.sh because that file (and the
# real-data/ tree it covers) was under an active claim of another lane
# when round 13 landed; this gate covers only the round-13 drivers.
#
# souc's process exit code is unreliable for both subcommands, so all
# verdicts are derived from captured stdout/stderr, never from $?.
# A per-file OK/FAIL line is printed; the gate exits 1 if any file fails.
#
# Runs from any cwd (all paths anchored at ROOT_DIR). Each `souc run`
# takes ~1-5 min (three GO cones resp. CL+UBERON, each with a full
# fixpoint and two ablation re-runs); a per-file timeout guards against
# hangs.
#
# Env overrides:
#   SOUC_BIN                          compiler wrapper (default: bin/souc)
#   ONTOLOGY_MULTI_RUN_TIMEOUT        per-file `souc run` timeout, seconds
#                                     (default: 900; ChEBI+PATO needs headroom)
#
# Exit 0 = all drivers pass. Exit 1 = at least one driver failed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
RUN_TIMEOUT="${ONTOLOGY_MULTI_RUN_TIMEOUT:-900}"

DRIVERS=(
    "artifacts/ontology-frontiers/multi-ontology/go_roots_elplus_driver.sio"
    "artifacts/ontology-frontiers/multi-ontology/obo_elplus_driver.sio"
    "artifacts/ontology-frontiers/multi-ontology/chebi_pato_elplus_driver.sio"
    "artifacts/ontology-frontiers/multi-ontology/open_fillers_elplus_driver.sio"
)

TMP_DIR="$(mktemp -d /tmp/ontology-multi-gate.XXXXXX)"
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
        printf '[ontology-multi] FAIL: %s (missing file: %s)\n' "$name" "$rel" >&2
        FAIL=$((FAIL + 1))
        return
    fi

    # (a) souc check — exit code unreliable, grep stdout.
    local check_log="$TMP_DIR/$name.check.log"
    "$SOUC_BIN" check "$src" >"$check_log" 2>&1 || true
    if grep -Fq 'parse error' "$check_log"; then
        printf '[ontology-multi]   check: parse error (see %s)\n' "$check_log" >&2
        problems=$((problems + 1))
    elif ! grep -Fq 'check: OK' "$check_log"; then
        printf '[ontology-multi]   check: missing "check: OK" (see %s)\n' "$check_log" >&2
        problems=$((problems + 1))
    fi

    # (b) souc run — exit code unreliable, require an exact ALL PASS line.
    local run_log="$TMP_DIR/$name.run.log"
    timeout "$RUN_TIMEOUT" "$SOUC_BIN" run "$src" >"$run_log" 2>&1 || true
    if ! grep -Fxq 'ALL PASS' "$run_log"; then
        printf '[ontology-multi]   run: missing exact "ALL PASS" line (see %s)\n' "$run_log" >&2
        problems=$((problems + 1))
    fi

    if [[ "$problems" -eq 0 ]]; then
        PASS=$((PASS + 1))
        printf '[ontology-multi] OK:   %s\n' "$rel"
    else
        FAIL=$((FAIL + 1))
        printf '[ontology-multi] FAIL: %s (%d problem(s))\n' "$rel" "$problems" >&2
    fi
}

echo "[ontology-multi] souc=$SOUC_BIN"
echo "[ontology-multi] drivers=${#DRIVERS[@]} run_timeout=${RUN_TIMEOUT}s"

if [[ ! -x "$SOUC_BIN" ]]; then
    echo "[ontology-multi] ERROR: souc wrapper not executable: $SOUC_BIN" >&2
    exit 1
fi

for rel in "${DRIVERS[@]}"; do
    gate_file "$rel"
done

printf '[ontology-multi] results: %d passed, %d failed\n' "$PASS" "$FAIL"

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
