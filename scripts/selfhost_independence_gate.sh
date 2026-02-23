#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/target/release/souc}"
BOOTSTRAP_MANIFEST="${SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST:-bootstrap/selfhost-kernel.manifest}"
STRICT_MODULE_GATING="${SOUNIO_SELFHOST_STRICT_MODULE_GATING:-1}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-independence-gate}"
LOG_DIR="$WORK_DIR/logs"
BOOTSTRAP_BUNDLE_DIR="${BOOTSTRAP_BUNDLE_DIR:-bootstrap}"
BOOTSTRAP_STATE_DIR="${BOOTSTRAP_STATE_DIR:-$WORK_DIR/bootstrap-state}"
INDEPENDENCE_CONTRACT_PATH="${INDEPENDENCE_CONTRACT_PATH:-benchmarks/independence/contract.v1.json}"
REQUIRE_SIGNED_POLICY="${SOUNIO_REQUIRE_SIGNED_POLICY:-0}"
POLICY_PATH="${SOUNIO_OPT_POLICY_PATH:-bootstrap/policies/policy.v1.json}"
DECISION_TRAIL_REQUIRED="${SOUNIO_OPT_DECISION_TRAIL_REQUIRED:-1}"
DECISION_TRAIL_PATH="${SOUNIO_OPT_DECISION_TRAIL_PATH:-$LOG_DIR/decision_trail.jsonl}"

mkdir -p "$LOG_DIR"

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
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

echo "SELFHOST_INDEPENDENCE_GATE_START"
echo "root=$ROOT_DIR"
echo "manifest=$BOOTSTRAP_MANIFEST"
echo "logs=$LOG_DIR"
echo "decision_trail_required=$SOUNIO_OPT_DECISION_TRAIL_REQUIRED"
echo "decision_trail_path=$SOUNIO_OPT_DECISION_TRAIL_PATH"

run_step "01-cargo-check-bins" cargo check -p souc --bins
run_step "02-cargo-test-lib" cargo test -p souc --lib
run_step "03-cargo-build-release" cargo build -p souc --release

if [ ! -x "$SOUC_BIN" ]; then
  echo "error: expected release binary not found: $SOUC_BIN" >&2
  exit 1
fi

run_step "04-bootstrap-verify" "$SOUC_BIN" bootstrap verify --bundle "$BOOTSTRAP_BUNDLE_DIR"
run_step "05-bootstrap-init" "$SOUC_BIN" bootstrap init --bundle "$BOOTSTRAP_BUNDLE_DIR" --state "$BOOTSTRAP_STATE_DIR"
run_step "06-bootstrap-cycle" "$SOUC_BIN" bootstrap cycle --state "$BOOTSTRAP_STATE_DIR"

run_step "06a-independence-benchmark-contract" env \
  SOUC_BIN="$SOUC_BIN" \
  CONTRACT_PATH="$INDEPENDENCE_CONTRACT_PATH" \
  bash scripts/independence_benchmark_gate.sh

run_step "06b-opt-policy-status" "$SOUC_BIN" opt policy status --policy "$POLICY_PATH"
if [[ "$REQUIRE_SIGNED_POLICY" == "1" ]]; then
  run_step "06c-opt-policy-eval" "$SOUC_BIN" opt policy eval --policy "$POLICY_PATH"
else
  echo "==> 06c-opt-policy-eval (skipped, set SOUNIO_REQUIRE_SIGNED_POLICY=1 to enforce)"
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
  "$SOUC_BIN" run self-hosted/ --check-only \
  2>&1 | tee "$RUNTIME_LOG"
RUNTIME_RC=${PIPESTATUS[0]}
set -e

if [[ "$RUNTIME_RC" -eq 124 || "$RUNTIME_RC" -eq 137 ]]; then
  echo "error: runtime fail-fast check timed out (possible local rebuild hang)" >&2
  exit 1
fi

if [[ "$RUNTIME_RC" -ne 0 ]]; then
  if rg -n "SELFHOST_BOOTSTRAP_ARTIFACTS_MISSING|SELFHOST_SIGNED_HARNESS_REQUIRED|SELFHOST_SIGNED_HARNESS_MISSING|SELFHOST_STRICT_DRIVER_FAILURE|SELFHOST_SEED_ONLY_ENFORCED" "$RUNTIME_LOG" >/dev/null 2>&1; then
    echo "SELFHOST_INDEPENDENCE_GATE_INFO runtime_fail_fast_status=expected_error rc=$RUNTIME_RC"
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
