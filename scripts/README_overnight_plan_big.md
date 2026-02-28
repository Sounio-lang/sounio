# Overnight Plan BIG Ops Notes

This document describes the reliability contract for the overnight Plan BIG lane.

## Core Commands

- Start runner:
  - `bash scripts/start_overnight_plan_big.sh --interval-sec 900 --max-runs 0 --stop-on-pass 0`
- Stop runner:
  - `bash scripts/stop_overnight_plan_big.sh`
- Inspect status:
  - `bash scripts/overnight_plan_big_status.sh --json`
- Health check (strict contract):
  - `bash scripts/check_overnight_plan_big_health.sh --auto-heal --tail-lines 20`
- Burn-in:
  - `bash scripts/overnight_plan_big_burnin.sh --duration-sec 86400 --check-interval-sec 60 --auto-heal 1`

## Strict Health Contract

Health is considered `healthy=true` only when all checks pass:

1. `state_running`
2. `runner_lock_consistent`
3. `heartbeat_valid_non_stale`
4. `latest_valid`
5. `latest_pass`

The `latest_pass` check requires `latest.status=pass`, `latest.rc=0`, and `latest.pass_marker=true`.

## Startup Modes

- `PLAN_BIG_OVERNIGHT_STARTUP_REQUIRE_FIRST_RESULT=1` (default):
  start only succeeds after `latest.v1.json` is refreshed by the new runner PID.
- `PLAN_BIG_OVERNIGHT_STARTUP_TIMEOUT_SEC=15` controls base handshake timeout.
- `PLAN_BIG_OVERNIGHT_FIRST_RESULT_TIMEOUT_SEC=900` controls first-result timeout.

## Burn-in Artifact

`artifacts/omega/overnight_plan_big_burnin.v1.json` schema:

- `duration_target_sec`
- `duration_actual_sec`
- `checks_total`
- `checks_passed`
- `first_failure_at_utc` (nullable)
- `status` (`pass|fail`)
- `checks` (per-cycle evidence)

## Gate Integration

`scripts/plan_big_gate.sh` now supports:

- `--require-overnight-burnin`
- `--no-require-overnight-burnin`

When burn-in is required, gate fails if the burn-in artifact is missing or not `status=pass`.
