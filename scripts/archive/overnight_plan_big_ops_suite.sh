#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WITH_GATE=0
RUNNER_GATE_SCRIPT="${PLAN_BIG_OPS_SUITE_RUNNER_GATE_SCRIPT:-/bin/true}"
RUNNER_GATE_ARGS="${PLAN_BIG_OPS_SUITE_RUNNER_GATE_ARGS:-}"
BURNIN_DURATION_SEC="${PLAN_BIG_OPS_SUITE_BURNIN_DURATION_SEC:-120}"
BURNIN_CHECK_INTERVAL_SEC="${PLAN_BIG_OPS_SUITE_BURNIN_INTERVAL_SEC:-20}"
TAIL_LINES="${PLAN_BIG_OPS_SUITE_TAIL_LINES:-20}"

usage() {
  cat <<USAGE
Usage: scripts/overnight_plan_big_ops_suite.sh [--with-gate] [--runner-gate-script PATH] [--runner-gate-args "ARGS"] [--burnin-duration-sec N] [--burnin-check-interval-sec N]

Options:
  --with-gate                   Include plan_big_gate strict call in suite
  --runner-gate-script PATH     Gate script used by overnight runner during suite (default: /bin/true)
  --runner-gate-args "ARGS"     Gate args string for overnight runner (default: empty)
  --burnin-duration-sec N       Burn-in duration for suite (default: 120)
  --burnin-check-interval-sec N Burn-in interval for suite (default: 20)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-gate)
      WITH_GATE=1
      shift
      ;;
    --runner-gate-script)
      RUNNER_GATE_SCRIPT="$2"
      shift 2
      ;;
    --runner-gate-args)
      RUNNER_GATE_ARGS="$2"
      shift 2
      ;;
    --burnin-duration-sec)
      BURNIN_DURATION_SEC="$2"
      shift 2
      ;;
    --burnin-check-interval-sec)
      BURNIN_CHECK_INTERVAL_SEC="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$BURNIN_DURATION_SEC" =~ ^[0-9]+$ ]] || [[ "$BURNIN_DURATION_SEC" -eq 0 ]]; then
  echo "error: burn-in duration must be positive integer" >&2
  exit 2
fi
if ! [[ "$BURNIN_CHECK_INTERVAL_SEC" =~ ^[0-9]+$ ]] || [[ "$BURNIN_CHECK_INTERVAL_SEC" -eq 0 ]]; then
  echo "error: burn-in check interval must be positive integer" >&2
  exit 2
fi
if [[ "$RUNNER_GATE_SCRIPT" == */* ]]; then
  if [[ "$RUNNER_GATE_SCRIPT" != /* ]]; then
    RUNNER_GATE_SCRIPT="$ROOT_DIR/$RUNNER_GATE_SCRIPT"
  fi
  if [[ ! -e "$RUNNER_GATE_SCRIPT" ]]; then
    echo "error: runner gate script not found: $RUNNER_GATE_SCRIPT" >&2
    exit 2
  fi
else
  if command -v "$RUNNER_GATE_SCRIPT" >/dev/null 2>&1; then
    RUNNER_GATE_SCRIPT="$(command -v "$RUNNER_GATE_SCRIPT")"
  else
    echo "error: runner gate command not found in PATH: $RUNNER_GATE_SCRIPT" >&2
    exit 2
  fi
fi

echo "[ops-suite] syntax check"
bash -n scripts/overnight_plan_big_runner.sh \
  scripts/start_overnight_plan_big.sh \
  scripts/stop_overnight_plan_big.sh \
  scripts/overnight_plan_big_status.sh \
  scripts/check_overnight_plan_big_health.sh \
  scripts/overnight_plan_big_burnin.sh \
  scripts/plan_big_strict_canary.sh \
  scripts/rotate_overnight_plan_big_artifacts.sh \
  scripts/overnight_plan_big_hourly_report.sh \
  scripts/tmux_big_ops_default.sh

echo "[ops-suite] ensure clean stop"
bash scripts/stop_overnight_plan_big.sh --force || true

echo "[ops-suite] start runner"
PLAN_BIG_OVERNIGHT_GATE_SCRIPT="$RUNNER_GATE_SCRIPT" \
PLAN_BIG_OVERNIGHT_GATE_ARGS="$RUNNER_GATE_ARGS" \
bash scripts/start_overnight_plan_big.sh --interval-sec 900 --max-runs 0 --stop-on-pass 0

echo "[ops-suite] status running"
bash scripts/overnight_plan_big_status.sh --json | jq -e '.state == "running"' >/dev/null

echo "[ops-suite] health contract"
bash scripts/check_overnight_plan_big_health.sh --tail-lines "$TAIL_LINES"

echo "[ops-suite] burn-in short soak"
bash scripts/overnight_plan_big_burnin.sh \
  --duration-sec "$BURNIN_DURATION_SEC" \
  --check-interval-sec "$BURNIN_CHECK_INTERVAL_SEC" \
  --tail-lines "$TAIL_LINES" \
  --auto-heal 1 \
  --fail-fast 1

echo "[ops-suite] hourly report"
if [[ "$WITH_GATE" -eq 1 ]]; then
  bash scripts/overnight_plan_big_hourly_report.sh
else
  PLAN_BIG_STRICT_CANARY_ENABLE=0 bash scripts/overnight_plan_big_hourly_report.sh
fi
jq -e '.schema == "sounio.plan.big.overnight.hourly-report.v1" and .snapshot.strict_canary_status != null and .snapshot.retention_status != null' artifacts/omega/overnight_plan_big_hourly_report.v1.json >/dev/null

if [[ "$WITH_GATE" -eq 1 ]]; then
  echo "[ops-suite] strict plan_big_gate with burn-in"
  bash scripts/plan_big_gate.sh --with-overnight-health --overnight-auto-heal --require-overnight-burnin
fi

echo "[ops-suite] stop runner"
bash scripts/stop_overnight_plan_big.sh --force || true

echo "OVERNIGHT_PLAN_BIG_OPS_SUITE_PASS"
