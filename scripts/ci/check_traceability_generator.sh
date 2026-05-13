#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(bash scripts/latest_diagnostic_run_dir.sh 2>/dev/null || true)"
fi

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "No diagnostic run dir found under artifacts/diagnostic/*"
  echo "Run scripts/fast_gate.sh (or pass a run dir) before checking traceability generation."
  exit 1
fi

bash scripts/paper/generate_release_traceability_report.sh "$RUN_DIR"

REPORT="$RUN_DIR/TRACEABILITY_MATRIX.md"
if [[ ! -f "$REPORT" ]]; then
  echo "Traceability report was not generated: $REPORT"
  exit 1
fi

if grep -n '\bn/a\b' "$REPORT" >/dev/null 2>&1; then
  echo "Traceability report contains unresolved evidence markers (n/a): $REPORT"
  grep -n '\bn/a\b' "$REPORT"
  exit 1
fi

echo "Traceability generator check passed: $RUN_DIR"
