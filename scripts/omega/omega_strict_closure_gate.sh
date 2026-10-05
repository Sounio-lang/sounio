#!/usr/bin/env bash
set -euo pipefail
# Resolved here, at the top, because this script changes directory later and a
# relative BASH_SOURCE stops resolving once it does.
_SOUC_GUARD_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/souc_verb_guard.sh"
. "$_SOUC_GUARD_LIB"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/souc}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-strict-closure-gate}"
LOG_DIR="$WORK_DIR/logs"
REPORT_PATH="${REPORT_PATH:-$ROOT_DIR/artifacts/omega/strict_no_rust_closure_gate.v1.json}"

BOOTSTRAP_BUNDLE_DIR="${BOOTSTRAP_BUNDLE_DIR:-bootstrap}"
BOOTSTRAP_STATE_DIR="${BOOTSTRAP_STATE_DIR:-$ROOT_DIR/.sounio-state}"
DRIVER_HARNESS_CACHE_DIR="${DRIVER_HARNESS_CACHE_DIR:-$BOOTSTRAP_STATE_DIR/driver_harness_cache}"
PREWARM_TARGET="${PREWARM_TARGET:-examples/minimal.sio}"

SEED_BUILD_SKIP_BUILD="${SEED_BUILD_SKIP_BUILD:-1}"
SEED_BUILD_PROFILE="${SEED_BUILD_PROFILE:-core}"
SEED_PATH="${SEED_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
SEED_SHA_PATH="${SEED_SHA_PATH:-${SEED_PATH}.sha256}"
BOOTSTRAP_SEED_TRUSTED_KEY="${BOOTSTRAP_SEED_TRUSTED_KEY:-sounio-dev}"

SMOKE_SOURCE="${SMOKE_SOURCE:-tests/run-pass/basic_math.sio}"
SMOKE_BIN="${SMOKE_BIN:-/tmp/sounio-strict-closure-basic_math}"
SMOKE_EXPECTED_OUTPUT="${SMOKE_EXPECTED_OUTPUT:-84}"
SMOKE_TIMEOUT_SECS="${SMOKE_TIMEOUT_SECS:-120}"

STATE_VERIFY_LOG="$LOG_DIR/01_bootstrap_verify.log"
STATE_INIT_LOG="$LOG_DIR/02_bootstrap_init.log"
STATE_CYCLE_LOG="$LOG_DIR/03_bootstrap_cycle.log"
PREWARM_LOG="$LOG_DIR/04_driver_harness_prewarm.log"
SEED_BUILD_LOG="$LOG_DIR/05_strict_seed_build.log"
SMOKE_COMPILE_LOG="$LOG_DIR/06_seed_enforced_compile.log"
SMOKE_RUN_LOG="$LOG_DIR/07_native_smoke_run.log"

STATUS="fail"
FAILURE_CLASS="unknown"
FAILURE_DETAIL=""
CHECKPOINT_BOOTSTRAP_STATE="pending"
CHECKPOINT_DRIVER_HARNESS="pending"
CHECKPOINT_STRICT_SEED_BUILD="pending"
CHECKPOINT_SEED_ENFORCED_SMOKE="pending"

mkdir -p "$LOG_DIR" "$(dirname "$REPORT_PATH")" "$DRIVER_HARNESS_CACHE_DIR"

run_with_timeout() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
    return $?
  fi

  "$@"
}

run_step() {
  local log_file="$1"
  shift
  set +e
  "$@" >"$log_file" 2>&1
  local rc=$?
  set -e
  return "$rc"
}

write_report() {
  python3 - "$REPORT_PATH" <<'PY'
import json
import os
import pathlib

report_path = pathlib.Path(os.environ["REPORT_PATH"])
payload = {
    "schema": "sounio.strict_closure_gate.v1",
    "status": os.environ["STATUS"],
    "failure_class": os.environ["FAILURE_CLASS"],
    "failure_detail": os.environ["FAILURE_DETAIL"],
    "checkpoints": {
        "bootstrap_state": os.environ["CHECKPOINT_BOOTSTRAP_STATE"],
        "driver_harness": os.environ["CHECKPOINT_DRIVER_HARNESS"],
        "strict_seed_build": os.environ["CHECKPOINT_STRICT_SEED_BUILD"],
        "seed_enforced_smoke": os.environ["CHECKPOINT_SEED_ENFORCED_SMOKE"],
    },
    "paths": {
        "souc_bin": os.environ["SOUC_BIN"],
        "bootstrap_state_dir": os.environ["BOOTSTRAP_STATE_DIR"],
        "driver_harness_cache_dir": os.environ["DRIVER_HARNESS_CACHE_DIR"],
        "seed_path": os.environ["SEED_PATH"],
    },
    "logs": {
        "bootstrap_verify": os.environ["STATE_VERIFY_LOG"],
        "bootstrap_init": os.environ["STATE_INIT_LOG"],
        "bootstrap_cycle": os.environ["STATE_CYCLE_LOG"],
        "driver_harness_prewarm": os.environ["PREWARM_LOG"],
        "strict_seed_build": os.environ["SEED_BUILD_LOG"],
        "seed_enforced_compile": os.environ["SMOKE_COMPILE_LOG"],
        "native_smoke_run": os.environ["SMOKE_RUN_LOG"],
    },
}
report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

finish() {
  export REPORT_PATH STATUS FAILURE_CLASS FAILURE_DETAIL
  export CHECKPOINT_BOOTSTRAP_STATE CHECKPOINT_DRIVER_HARNESS
  export CHECKPOINT_STRICT_SEED_BUILD CHECKPOINT_SEED_ENFORCED_SMOKE
  export SOUC_BIN BOOTSTRAP_STATE_DIR DRIVER_HARNESS_CACHE_DIR SEED_PATH
  export STATE_VERIFY_LOG STATE_INIT_LOG STATE_CYCLE_LOG PREWARM_LOG
  export SEED_BUILD_LOG SMOKE_COMPILE_LOG SMOKE_RUN_LOG

  write_report

  echo "STRICT_CLOSURE_GATE_STATUS status=$STATUS failure_class=$FAILURE_CLASS"
  echo "report=$REPORT_PATH"

  if [[ "$STATUS" == "pass" ]]; then
    exit 0
  fi
  exit 1
}

echo "STRICT_CLOSURE_GATE_START"
echo "souc_bin=$SOUC_BIN"
echo "state_dir=$BOOTSTRAP_STATE_DIR"
echo "driver_harness_cache_dir=$DRIVER_HARNESS_CACHE_DIR"
echo "seed_path=$SEED_PATH"
echo "smoke_source=$SMOKE_SOURCE"

sounio_require_souc

# Checkpoint 1: signed bootstrap state available.
# Refuse before the work, and name what is actually missing: the `bootstrap`
# verbs went with the Rust crate (79acc192e1) and the fall-through
# diagnostic reports a missing FILE. See scripts/lib/souc_verb_guard.sh.
require_souc_verb "$SOUC_BIN" bootstrap "the signed-bootstrap-state checkpoint"
if ! run_step "$STATE_VERIFY_LOG" "$SOUC_BIN" bootstrap verify --bundle "$BOOTSTRAP_BUNDLE_DIR"; then
  CHECKPOINT_BOOTSTRAP_STATE="fail"
  FAILURE_CLASS="bootstrap_verify_failed"
  FAILURE_DETAIL="bootstrap verify failed"
  finish
fi
if ! run_step "$STATE_INIT_LOG" "$SOUC_BIN" bootstrap init --bundle "$BOOTSTRAP_BUNDLE_DIR" --state "$BOOTSTRAP_STATE_DIR"; then
  CHECKPOINT_BOOTSTRAP_STATE="fail"
  FAILURE_CLASS="bootstrap_init_failed"
  FAILURE_DETAIL="bootstrap init failed"
  finish
fi
if ! run_step "$STATE_CYCLE_LOG" "$SOUC_BIN" bootstrap cycle --state "$BOOTSTRAP_STATE_DIR"; then
  CHECKPOINT_BOOTSTRAP_STATE="fail"
  FAILURE_CLASS="bootstrap_cycle_failed"
  FAILURE_DETAIL="bootstrap cycle failed"
  finish
fi
CHECKPOINT_BOOTSTRAP_STATE="pass"

# Seed driver harness cache from bootstrap bundle seeds (if not already populated).
# The bundle includes driver_harness_*.sobc alongside the seed bytecode.
# This must happen after bootstrap init so DRIVER_HARNESS_CACHE_DIR exists.
if ! ls "$DRIVER_HARNESS_CACHE_DIR"/driver_harness_*.sobc >/dev/null 2>&1; then
  if ls "$BOOTSTRAP_BUNDLE_DIR/seeds"/driver_harness_*.sobc >/dev/null 2>&1; then
    cp "$BOOTSTRAP_BUNDLE_DIR/seeds"/driver_harness_*.sobc "$DRIVER_HARNESS_CACHE_DIR/"
    echo "STRICT_CLOSURE_GATE driver_harness seeded from bundle seeds: $(ls "$DRIVER_HARNESS_CACHE_DIR"/driver_harness_*.sobc)"
  fi
fi

# Checkpoint 2: driver harness available.
if ! run_step "$PREWARM_LOG" env \
  SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR="$DRIVER_HARNESS_CACHE_DIR" \
  "$SOUC_BIN" run "$PREWARM_TARGET" --use-sounio-compiler --check-only; then
  CHECKPOINT_DRIVER_HARNESS="fail"
  if rg -n "SELFHOST_DRIVER_HARNESS_UNAVAILABLE" "$PREWARM_LOG" >/dev/null 2>&1; then
    FAILURE_CLASS="driver_harness_unavailable"
    FAILURE_DETAIL="driver harness unavailable during prewarm"
  else
    FAILURE_CLASS="driver_harness_prewarm_failed"
    FAILURE_DETAIL="driver harness prewarm failed"
  fi
  finish
fi
if ! ls "$DRIVER_HARNESS_CACHE_DIR"/driver_harness_*.sobc >/dev/null 2>&1; then
  CHECKPOINT_DRIVER_HARNESS="fail"
  FAILURE_CLASS="driver_harness_cache_missing"
  FAILURE_DETAIL="prewarm succeeded but harness cache artifacts missing"
  finish
fi
CHECKPOINT_DRIVER_HARNESS="pass"

# Checkpoint 3: strict seed build (no fallback).
if ! run_step "$SEED_BUILD_LOG" env \
  SKIP_BUILD="$SEED_BUILD_SKIP_BUILD" \
  BOOTSTRAP_PROFILE="$SEED_BUILD_PROFILE" \
  SEED_FALLBACK_ENABLED=0 \
  bash "$ROOT_DIR/scripts/build_bootstrap_seed.sh"; then
  CHECKPOINT_STRICT_SEED_BUILD="fail"
  if rg -n "SELFHOST_BOOTSTRAP_ARTIFACTS_MISSING" "$SEED_BUILD_LOG" >/dev/null 2>&1; then
    FAILURE_CLASS="selfhost_bootstrap_artifacts_missing"
    FAILURE_DETAIL="strict seed build failed: missing bootstrap artifacts"
  elif rg -n "SELFHOST_DRIVER_HARNESS_UNAVAILABLE" "$SEED_BUILD_LOG" >/dev/null 2>&1; then
    FAILURE_CLASS="driver_harness_unavailable"
    FAILURE_DETAIL="strict seed build failed: harness unavailable"
  else
    FAILURE_CLASS="strict_seed_build_failed"
    FAILURE_DETAIL="strict seed build failed"
  fi
  finish
fi
if [[ ! -f "$SEED_PATH" || ! -f "$SEED_SHA_PATH" ]]; then
  CHECKPOINT_STRICT_SEED_BUILD="fail"
  FAILURE_CLASS="strict_seed_artifact_missing"
  FAILURE_DETAIL="strict seed build completed without seed artifacts"
  finish
fi
CHECKPOINT_STRICT_SEED_BUILD="pass"

# Checkpoint 4: seed-enforced dispatcher/native smoke.
seed_digest="$(sha256sum "$SEED_PATH" | awk '{print $1}')"
seed_sig_tmp="$WORK_DIR/seed_gate.sig"
printf 'SOUNIO-SEED-SIG-V1 key=%s sha256=%s\n' "$BOOTSTRAP_SEED_TRUSTED_KEY" "$seed_digest" >"$seed_sig_tmp"

if ! run_step "$SMOKE_COMPILE_LOG" env \
  SOUNIO_BOOTSTRAP_SEED_PATH="$SEED_PATH" \
  SOUNIO_BOOTSTRAP_SEED_SHA256_PATH="$SEED_SHA_PATH" \
  SOUNIO_BOOTSTRAP_SEED_SIG_PATH="$seed_sig_tmp" \
  SOUNIO_BOOTSTRAP_SEED_ENFORCE=1 \
  SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY="$BOOTSTRAP_SEED_TRUSTED_KEY" \
  SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST=bootstrap/selfhost-kernel.manifest \
  "$SOUC_BIN" run self-hosted/ -- compile --backend=native -o "$SMOKE_BIN" "$SMOKE_SOURCE"; then
  CHECKPOINT_SEED_ENFORCED_SMOKE="fail"
  FAILURE_CLASS="seed_enforced_compile_failed"
  FAILURE_DETAIL="seed-enforced native compile failed"
  finish
fi
if [[ ! -x "$SMOKE_BIN" ]]; then
  CHECKPOINT_SEED_ENFORCED_SMOKE="fail"
  FAILURE_CLASS="seed_enforced_binary_missing"
  FAILURE_DETAIL="native binary missing after compile"
  finish
fi

set +e
run_with_timeout "$SMOKE_TIMEOUT_SECS" "$SMOKE_BIN" >"$SMOKE_RUN_LOG" 2>&1
smoke_rc=$?
set -e
if [[ "$smoke_rc" -ne 0 ]]; then
  CHECKPOINT_SEED_ENFORCED_SMOKE="fail"
  FAILURE_CLASS="native_smoke_runtime_failed"
  FAILURE_DETAIL="native smoke binary exited non-zero"
  finish
fi

smoke_output="$(tr -d '\r' <"$SMOKE_RUN_LOG" | tr -d '\n' | sed 's/[[:space:]]\\+$//')"
if [[ "$smoke_output" != "$SMOKE_EXPECTED_OUTPUT" ]]; then
  CHECKPOINT_SEED_ENFORCED_SMOKE="fail"
  FAILURE_CLASS="native_smoke_output_mismatch"
  FAILURE_DETAIL="native smoke output mismatch"
  finish
fi

CHECKPOINT_SEED_ENFORCED_SMOKE="pass"
STATUS="pass"
FAILURE_CLASS=""
FAILURE_DETAIL=""
finish
