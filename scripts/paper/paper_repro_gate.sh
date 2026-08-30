#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"
OUT_JSON="${PAPER_REPRO_GATE_STATUS_JSON:-$ROOT_DIR/artifacts/omega/paper_repro_gate_status.v1.json}"

RUN_REPRO=1
STATUS_JSON=""

usage() {
  cat <<'EOF'
Usage: scripts/paper/paper_repro_gate.sh [--status-json PATH] [--no-run]

Options:
  --status-json PATH   Evaluate an existing status_summary.v1.json
  --no-run             Skip running paper/reproduce.sh
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --status-json)
      STATUS_JSON="$2"
      shift 2
      ;;
    --no-run)
      RUN_REPRO=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required for paper_repro_gate.sh" >&2
  exit 2
fi

if [[ "$RUN_REPRO" -eq 1 ]]; then
  echo "[paper-repro-gate] running paper/reproduce.sh"
  (cd "$PAPER_DIR" && ./reproduce.sh)
fi

if [[ -z "$STATUS_JSON" ]]; then
  STATUS_JSON="$(ls -1dt "$PAPER_DIR"/artifacts/*/status_summary.v1.json 2>/dev/null | head -n1 || true)"
fi

if [[ -z "$STATUS_JSON" || ! -f "$STATUS_JSON" ]]; then
  echo "error: status summary not found (expected paper/artifacts/*/status_summary.v1.json)" >&2
  exit 1
fi

source_rel="${STATUS_JSON#$ROOT_DIR/}"

critical_steps=(pdf_build bateman_convergence reference_check)
optional_steps=(python_benchmark julia_benchmark sounio_benchmark)

step_failed() {
  local step="$1"
  local status rc
  status="$(jq -r --arg s "$step" '.steps[$s].status // "missing"' "$STATUS_JSON")"
  rc="$(jq -r --arg s "$step" '.steps[$s].rc // -999999' "$STATUS_JSON")"
  [[ "$status" != "pass" || "$rc" != "0" ]]
}

print_step() {
  local step="$1"
  local status rc
  status="$(jq -r --arg s "$step" '.steps[$s].status // "missing"' "$STATUS_JSON")"
  rc="$(jq -r --arg s "$step" '.steps[$s].rc // -999999' "$STATUS_JSON")"
  echo "  - $step: status=$status rc=$rc"
}

critical_fail_count=0
optional_fail_count=0

for step in "${critical_steps[@]}"; do
  if step_failed "$step"; then
    critical_fail_count=$((critical_fail_count + 1))
  fi
done

for step in "${optional_steps[@]}"; do
  if step_failed "$step"; then
    optional_fail_count=$((optional_fail_count + 1))
  fi
done

overall_status="pass"
if [[ "$critical_fail_count" -gt 0 ]]; then
  overall_status="fail"
fi

mkdir -p "$(dirname "$OUT_JSON")"

critical_checks_json="$(jq '{pdf_build: .steps.pdf_build, bateman_convergence: .steps.bateman_convergence, reference_check: .steps.reference_check}' "$STATUS_JSON")"
optional_checks_json="$(jq '{python_benchmark: .steps.python_benchmark, julia_benchmark: .steps.julia_benchmark, sounio_benchmark: .steps.sounio_benchmark}' "$STATUS_JSON")"

jq -n \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_status_json "$source_rel" \
  --arg status "$overall_status" \
  --argjson critical_fail_count "$critical_fail_count" \
  --argjson optional_fail_count "$optional_fail_count" \
  --argjson critical_checks "$critical_checks_json" \
  --argjson optional_checks "$optional_checks_json" \
  '{
    schema: "sounio.paper.repro.gate-status.v1",
    generated_at_utc: $generated_at_utc,
    source_status_json: $source_status_json,
    status: $status,
    critical_fail_count: $critical_fail_count,
    optional_fail_count: $optional_fail_count,
    critical_checks: $critical_checks,
    optional_checks: $optional_checks
  }' >"$OUT_JSON"

echo "[paper-repro-gate] source_status_json=$source_rel"
echo "[paper-repro-gate] critical checks:"
for step in "${critical_steps[@]}"; do
  print_step "$step"
done
echo "[paper-repro-gate] optional checks:"
for step in "${optional_steps[@]}"; do
  print_step "$step"
done
echo "[paper-repro-gate] gate_status_json=${OUT_JSON#$ROOT_DIR/}"

if [[ "$overall_status" == "pass" ]]; then
  echo "PAPER_REPRO_GATE_PASS"
  exit 0
fi

echo "error: PAPER_REPRO_GATE_FAIL critical_fail_count=$critical_fail_count" >&2
exit 1
