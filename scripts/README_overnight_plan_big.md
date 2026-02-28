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
- Hourly report snapshot:
  - `bash scripts/overnight_plan_big_hourly_report.sh`
- Ops-only regression suite:
  - `bash scripts/overnight_plan_big_ops_suite.sh`
  - isolated default (recommended for infra-only checks): uses runner gate `/bin/true`
  - strict mode: `bash scripts/overnight_plan_big_ops_suite.sh --with-gate --runner-gate-script scripts/plan_big_gate.sh`

## Tmux Default Environment

Use the default ops cockpit (SSH-safe):

- Start or refresh environment:
  - `bash scripts/tmux_big_ops_default.sh up`
- Reset session + restart runner/burn-in (recommended when changing env vars):
  - `bash scripts/tmux_big_ops_default.sh up --reset`
- Attach:
  - `bash scripts/tmux_big_ops_default.sh attach`
- Quick status:
  - `bash scripts/tmux_big_ops_default.sh status`
- Stop session only:
  - `bash scripts/tmux_big_ops_default.sh down`
- Stop session and processes:
  - `bash scripts/tmux_big_ops_default.sh down --stop-processes`

The `up` command applies tmux defaults (`mouse`, high `history-limit`, renumbered windows, remain-on-exit panes) and ensures overnight runner + burn-in are active.
With `--reset`, it also stops existing runner/burn-in first, then starts clean (deterministic startup with current environment).
It also opens default windows: `status`, `health`, `gate`, `report`, `burnin-log`.
You can override runner gate command for tmux startup via:

- `PLAN_BIG_OVERNIGHT_GATE_SCRIPT` (default: `scripts/plan_big_gate.sh`)
- `PLAN_BIG_OVERNIGHT_GATE_ARGS` (default: empty)

The runner accepts both executable binaries (example: `/bin/true`) and shell scripts for `PLAN_BIG_OVERNIGHT_GATE_SCRIPT`.

Two practical startup profiles:

- Strict (default):
  - `bash scripts/tmux_big_ops_default.sh up --reset`
- Infra/liveness (keeps health green independent of gate failures):
  - `PLAN_BIG_OVERNIGHT_GATE_SCRIPT=/bin/true bash scripts/tmux_big_ops_default.sh up --reset`

Shortcut wrappers:

- Infra profile wrapper:
  - `bash scripts/tmux_big_ops_infra.sh up --reset`
- Strict profile wrapper:
  - `bash scripts/tmux_big_ops_strict.sh up --reset`

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
