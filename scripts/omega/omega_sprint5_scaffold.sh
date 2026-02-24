#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_PATH="${1:-artifacts/omega/sprint_5_0_scaffold.json}"
GATE_LOG="artifacts/omega/sprint_4_0_gate_full.log"
GENESIS_MANIFEST="artifacts/omega/omega_genesis.v1.0.json"
BASELINE_FREEZE="artifacts/omega/baseline_freeze.v1.json"
TODOS="docs/SELFHOST_OMEGA_TODOS.md"

for path in "$GATE_LOG" "$GENESIS_MANIFEST" "$BASELINE_FREEZE" "$TODOS"; do
  if [ ! -f "$path" ]; then
    echo "error: required baseline artifact missing: $path" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$OUT_PATH")"

sha_gate="$(sha256sum "$GATE_LOG" | cut -d' ' -f1)"
sha_genesis="$(sha256sum "$GENESIS_MANIFEST" | cut -d' ' -f1)"
sha_freeze="$(sha256sum "$BASELINE_FREEZE" | cut -d' ' -f1)"
sha_todos="$(sha256sum "$TODOS" | cut -d' ' -f1)"

generated_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > "$OUT_PATH" <<JSON
{
  "schema": "sounio.omega.sprint5-scaffold.v1",
  "generated_at_utc": "$generated_at_utc",
  "locked_baseline": {
    "sprint4_gate_log": {
      "path": "$GATE_LOG",
      "sha256": "$sha_gate"
    },
    "genesis_manifest": {
      "path": "$GENESIS_MANIFEST",
      "sha256": "$sha_genesis"
    },
    "baseline_freeze": {
      "path": "$BASELINE_FREEZE",
      "sha256": "$sha_freeze"
    },
    "todos": {
      "path": "$TODOS",
      "sha256": "$sha_todos"
    }
  },
  "tracks": [
    {
      "id": "s5-track-1",
      "title": "Merkle inclusion proof hardening",
      "status": "planned"
    },
    {
      "id": "s5-track-2",
      "title": "QIR emitter replay determinism",
      "status": "planned"
    },
    {
      "id": "s5-track-3",
      "title": "Hardware trend anomaly envelope",
      "status": "planned"
    }
  ],
  "entry_commands": [
    "bash scripts/omega/omega_sprint5_scaffold.sh",
    "bash scripts/omega_sprint1_gate.sh --strict --report-full",
    "python3 scripts/omega/omega_weekly_drift_report.py --window 7"
  ],
  "notes": "Scaffold only. No strict-gate behavior changes in this step."
}
JSON

echo "omega_sprint5_scaffold: out=$OUT_PATH gate_sha256=$sha_gate genesis_sha256=$sha_genesis"
