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
BOOTSTRAP_MANIFEST="${SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST:-bootstrap/selfhost-kernel.manifest}"
STRICT_MODULE_GATING="${SOUNIO_SELFHOST_STRICT_MODULE_GATING:-1}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-independence-gate}"
LOG_DIR="$WORK_DIR/logs"
BOOTSTRAP_BUNDLE_DIR="${BOOTSTRAP_BUNDLE_DIR:-bootstrap}"
BOOTSTRAP_STATE_DIR="${BOOTSTRAP_STATE_DIR:-$WORK_DIR/bootstrap-state}"
INDEPENDENCE_CONTRACT_PATH="${INDEPENDENCE_CONTRACT_PATH:-benchmarks/independence/contract.v1.json}"
DRIVER_HARNESS_CACHE_DIR="${DRIVER_HARNESS_CACHE_DIR:-$WORK_DIR/driver_harness_cache}"
PREWARM_TARGET="${PREWARM_TARGET:-examples/minimal.sio}"
REQUIRE_SIGNED_POLICY="${SOUNIO_REQUIRE_SIGNED_POLICY:-0}"
POLICY_PATH="${SOUNIO_OPT_POLICY_PATH:-bootstrap/policies/policy.v1.json}"
DECISION_TRAIL_REQUIRED="${SOUNIO_OPT_DECISION_TRAIL_REQUIRED:-1}"
DECISION_TRAIL_PATH="${SOUNIO_OPT_DECISION_TRAIL_PATH:-$LOG_DIR/decision_trail.jsonl}"
POLICY_SMOKE_OUTPUT="${SOUNIO_POLICY_SMOKE_OUTPUT:-$WORK_DIR/policy_status_smoke.v2.json}"
POLICY_SMOKE_ENV_PATH="${SOUNIO_POLICY_SMOKE_ENV_PATH:-$LOG_DIR/policy_smoke.env}"
POLICY_CANONICAL_ENV_PATH="${OMEGA_CANONICAL_ENV_OUT:-$WORK_DIR/canonical_key.env}"
POLICY_STATUS_SCRIPT="${OMEGA_POLICY_STATUS_SCRIPT:-scripts/omega/omega_policy_status.sh}"

mkdir -p "$LOG_DIR" "$DRIVER_HARNESS_CACHE_DIR"

export SOUNIO_OPT_DECISION_TRAIL_REQUIRED="$DECISION_TRAIL_REQUIRED"
export SOUNIO_OPT_DECISION_TRAIL_PATH="$DECISION_TRAIL_PATH"

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
  local name="$1"
  shift
  echo "==> $name"
  set +e
  set -o pipefail
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  local rc=${PIPESTATUS[0]}
  set +o pipefail
  set -e
  return "$rc"
}

echo "SELFHOST_INDEPENDENCE_GATE_START"
echo "root=$ROOT_DIR"
echo "manifest=$BOOTSTRAP_MANIFEST"
echo "logs=$LOG_DIR"
echo "decision_trail_required=$SOUNIO_OPT_DECISION_TRAIL_REQUIRED"
echo "decision_trail_path=$SOUNIO_OPT_DECISION_TRAIL_PATH"
echo "driver_harness_cache_dir=$DRIVER_HARNESS_CACHE_DIR"
echo "prewarm_target=$PREWARM_TARGET"

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "==> 01-cargo-check-bins (disabled: repo-hard no-rust)"
  echo "==> 02-cargo-test-lib (disabled: repo-hard no-rust)"
  echo "==> 03-cargo-build-release (disabled: repo-hard no-rust)"
fi
sounio_require_souc
run_step "01-souc-version" "$SOUC_BIN" --version

if [ ! -x "$SOUC_BIN" ]; then
  echo "error: expected release binary not found: $SOUC_BIN" >&2
  exit 1
fi

# Refuse before the work, and name what is actually missing: the `bootstrap`
# verbs went with the Rust crate (79acc192e1) and the fall-through
# diagnostic reports a missing FILE. See scripts/lib/souc_verb_guard.sh.
require_souc_verb "$SOUC_BIN" bootstrap "signed bootstrap bundle verify/init/cycle"
run_step "04-bootstrap-verify" "$SOUC_BIN" bootstrap verify --bundle "$BOOTSTRAP_BUNDLE_DIR"
run_step "05-bootstrap-init" "$SOUC_BIN" bootstrap init --bundle "$BOOTSTRAP_BUNDLE_DIR" --state "$BOOTSTRAP_STATE_DIR"
run_step "06-bootstrap-cycle" "$SOUC_BIN" bootstrap cycle --state "$BOOTSTRAP_STATE_DIR"
echo "==> 06a-driver-harness-prewarm"
PREWARM_LOG="$LOG_DIR/06a-driver-harness-prewarm.log"
set +e
run_with_timeout 180 env \
  SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR="$DRIVER_HARNESS_CACHE_DIR" \
  "$SOUC_BIN" run "$PREWARM_TARGET" --use-sounio-compiler --check-only \
  2>&1 | tee "$PREWARM_LOG"
PREWARM_RC=${PIPESTATUS[0]}
set -e
if [[ "$PREWARM_RC" -eq 0 ]]; then
  echo "SELFHOST_INDEPENDENCE_GATE_INFO driver_harness_prewarm_status=ok cache_dir=$DRIVER_HARNESS_CACHE_DIR"
elif rg -n "SELFHOST_DRIVER_HARNESS_UNAVAILABLE" "$PREWARM_LOG" >/dev/null 2>&1; then
  echo "SELFHOST_INDEPENDENCE_GATE_INFO driver_harness_prewarm_status=blocked reason=SELFHOST_DRIVER_HARNESS_UNAVAILABLE rc=$PREWARM_RC cache_dir=$DRIVER_HARNESS_CACHE_DIR"
else
  echo "SELFHOST_INDEPENDENCE_GATE_INFO driver_harness_prewarm_status=warn rc=$PREWARM_RC cache_dir=$DRIVER_HARNESS_CACHE_DIR"
fi

run_step "06b-independence-benchmark-contract" env \
  SOUC_BIN="$SOUC_BIN" \
  CONTRACT_PATH="$INDEPENDENCE_CONTRACT_PATH" \
  bash scripts/independence_benchmark_gate.sh

if [ ! -x "$POLICY_STATUS_SCRIPT" ]; then
  echo "error: policy status script not executable: $POLICY_STATUS_SCRIPT" >&2
  exit 2
fi
run_step "06c-opt-policy-status" "$POLICY_STATUS_SCRIPT" \
  --policy "$POLICY_PATH" \
  --souc "$SOUC_BIN" \
  --corpus benchmarks/independence \
  --canonical-env "$POLICY_CANONICAL_ENV_PATH" \
  --smoke-env "$POLICY_SMOKE_ENV_PATH" \
  --smoke-out "$POLICY_SMOKE_OUTPUT"
if [[ "$REQUIRE_SIGNED_POLICY" == "1" ]]; then
  if [ -f "$POLICY_SMOKE_ENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$POLICY_SMOKE_ENV_PATH"
  fi
  POLICY_STATUS_PATH="${SOUNIO_POLICY_STATUS_PATH:-$POLICY_PATH}"
  run_step "06d-opt-policy-eval" env \
    SOUNIO_POLICY_VERIFY_KEY_PATH="${SOUNIO_POLICY_VERIFY_KEY_PATH:-}" \
    "$SOUC_BIN" opt policy eval --policy "$POLICY_STATUS_PATH"
else
  echo "==> 06d-opt-policy-eval (skipped, set SOUNIO_REQUIRE_SIGNED_POLICY=1 to enforce)"
fi

run_step "07-selfhost-strict-check" env \
  SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST="$BOOTSTRAP_MANIFEST" \
  SOUNIO_SELFHOST_STRICT_MODULE_GATING="$STRICT_MODULE_GATING" \
  "$SOUC_BIN" check self-hosted/

run_step "08-test-list" "$SOUC_BIN" test --list --format compact

echo "==> 09-runtime-fail-fast"
RUNTIME_LOG="$LOG_DIR/09-runtime-fail-fast.log"
set +e
run_with_timeout 180 env \
  SOUNIO_SELFHOST_EXECUTOR=native-driver \
  SOUNIO_SELFHOST_DRIVER_ALLOW_LOCAL_REBUILD=0 \
  SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR="$DRIVER_HARNESS_CACHE_DIR" \
  "$SOUC_BIN" run self-hosted/ --check-only \
  2>&1 | tee "$RUNTIME_LOG"
RUNTIME_RC=${PIPESTATUS[0]}
set -e

if [[ "$RUNTIME_RC" -eq 124 || "$RUNTIME_RC" -eq 137 ]]; then
  echo "error: runtime fail-fast check timed out (possible local rebuild hang)" >&2
  exit 1
fi

if [[ "$RUNTIME_RC" -ne 0 ]]; then
  if rg -n "SELFHOST_BOOTSTRAP_ARTIFACTS_MISSING|SELFHOST_SIGNED_HARNESS_REQUIRED|SELFHOST_SIGNED_HARNESS_MISSING|SELFHOST_STRICT_DRIVER_FAILURE|SELFHOST_SEED_ONLY_ENFORCED|SELFHOST_DRIVER_HARNESS_UNAVAILABLE" "$RUNTIME_LOG" >/dev/null 2>&1; then
    runtime_reason="expected_error"
    if rg -n "SELFHOST_DRIVER_HARNESS_UNAVAILABLE" "$RUNTIME_LOG" >/dev/null 2>&1; then
      runtime_reason="driver_harness_unavailable"
    fi
    echo "SELFHOST_INDEPENDENCE_GATE_INFO runtime_fail_fast_status=expected_error rc=$RUNTIME_RC reason=$runtime_reason"
  else
    echo "error: runtime fail-fast check returned unexpected failure (rc=$RUNTIME_RC)" >&2
    exit "$RUNTIME_RC"
  fi
fi

if rg -n "LEGACY_SELFHOST_ENV_REMOVED" "$RUNTIME_LOG" >/dev/null 2>&1; then
  echo "error: runtime fail-fast check emitted legacy env diagnostic (unexpected in no-rust cutover)" >&2
  exit 1
fi

if rg -n "SELFHOST=driver-first schema=v1 event=driver_orchestration .* status=fallback|SELFHOST=oracle backend=rust" "$RUNTIME_LOG" >/dev/null 2>&1; then
  echo "error: runtime fail-fast check observed forbidden rust fallback/oracle markers" >&2
  exit 1
fi

echo "SELFHOST_INDEPENDENCE_GATE_PASS"
