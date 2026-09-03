#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

EXPLICIT_SOUC_BIN="${SOUC_BIN:-}"
SOUC_BIN="${SOUC_BIN:-./souc}"
if [ -z "$EXPLICIT_SOUC_BIN" ] && [ -x "$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh" ]; then
  SOUC_BIN="$(
    OMEGA_SOUC_REQUIRE_PINNED="${OMEGA_SOUC_REQUIRE_PINNED:-1}" \
    OMEGA_SOUC_ALLOW_LOCAL_FALLBACK="${OMEGA_SOUC_ALLOW_LOCAL_FALLBACK:-0}" \
      "$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh" --print-path
  )"
fi
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-cycle-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"
BOOTSTRAP_BUNDLE_DIR="${BOOTSTRAP_BUNDLE_DIR:-bootstrap}"
BOOTSTRAP_STATE_BASE_DIR="${BOOTSTRAP_STATE_BASE_DIR:-$WORK_DIR/bootstrap-state}"

STAGE1_PATH="${STAGE1_PATH:-$ARTIFACT_DIR/selfhost.stage1.sobc}"
STAGE2_PATH="${STAGE2_PATH:-$ARTIFACT_DIR/selfhost.stage2.sobc}"
CACHE_PATH="${CACHE_PATH:-self-hosted/.sounio_bytecode.sobc}"

BUILD_TIMEOUT_SECS="${BUILD_TIMEOUT_SECS:-900}"
COMPILE_TIMEOUT_SECS="${COMPILE_TIMEOUT_SECS:-900}"
SKIP_BUILD="${SOUNIO_SELFHOST_CYCLE_SKIP_BUILD:-0}"
SEED_ENFORCE="${SOUNIO_SELFHOST_CYCLE_SEED_ENFORCE:-0}"
SEED_PATH="${SOUNIO_SELFHOST_CYCLE_SEED_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
SEED_SHA256_PATH="${SOUNIO_SELFHOST_CYCLE_SEED_SHA256_PATH:-${SEED_PATH}.sha256}"
SEED_SIG_PATH="${SOUNIO_SELFHOST_CYCLE_SEED_SIG_PATH:-${SEED_PATH}.sig}"
CYCLE_FORCE_DYNAMIC="${SOUNIO_SELFHOST_CYCLE_FORCE_DYNAMIC:-1}"
NO_RUST_MARKER_ENFORCE="${SOUNIO_SELFHOST_CYCLE_NO_RUST_MARKER_ENFORCE:-1}"
NO_RUST_MARKER_TIMEOUT_SECS="${SOUNIO_SELFHOST_CYCLE_NO_RUST_MARKER_TIMEOUT_SECS:-30}"
BOOTSTRAP_MANIFEST_PATH="${BOOTSTRAP_MANIFEST_PATH:-${SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST:-bootstrap/selfhost-kernel.manifest}}"
INDEPENDENCE_CONTRACT_PATH="${INDEPENDENCE_CONTRACT_PATH:-benchmarks/independence/contract.v1.json}"
BOOTSTRAP_MANIFEST_METADATA_PATH="${BOOTSTRAP_MANIFEST_METADATA_PATH:-bootstrap/artifacts/manifest.v2.json}"
BOOTSTRAP_SEED_TRUSTED_KEY="${SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY:-}"

resolve_bootstrap_seed_trusted_key() {
  if [[ -n "$BOOTSTRAP_SEED_TRUSTED_KEY" ]]; then
    echo "$BOOTSTRAP_SEED_TRUSTED_KEY"
    return 0
  fi

  if [[ -f "$BOOTSTRAP_MANIFEST_METADATA_PATH" ]]; then
    # Pure-Sounio key extraction (replaces python json.load["key_id"] heredoc).
    # ROOT_DIR in scripts/selfhost/*.sh resolves to scripts/, not repo root —
    # use ../bin/kretikos. --print-or-empty matches python's `payload.get(key, "")`.
    local manifest_key
    manifest_key="$("$ROOT_DIR/../bin/kretikos" kaxi-validate-evidence \
        "$BOOTSTRAP_MANIFEST_METADATA_PATH" --print-or-empty "key_id" 2>/dev/null || true)"
    if [[ -n "$manifest_key" ]]; then
      echo "$manifest_key"
      return 0
    fi
  fi

  echo "sounio-dev"
}

BOOTSTRAP_SEED_TRUSTED_KEY="$(resolve_bootstrap_seed_trusted_key)"

if [ "$CYCLE_FORCE_DYNAMIC" = "1" ]; then
  # Cycle reproducibility should remain testable even when a seed is present.
  # Force the bootstrap seed loader down the dynamic path for this gate.
  SEED_PATH="${SEED_PATH}.missing"
  SEED_SHA256_PATH="${SEED_SHA256_PATH}.missing"
  SEED_SIG_PATH="${SEED_SIG_PATH}.missing"
fi

run_with_timeout() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
    return $?
  fi

  # Pure-perl timeout fallback (replaces python3 subprocess.run heredoc).
  # See selfhost_zero_fallback_gate.sh for rationale.
  if command -v perl >/dev/null 2>&1; then
    perl -e 'alarm $ARGV[0]; shift @ARGV; exec @ARGV; exit 127' "$seconds" "$@"
    local rc=$?
    [[ $rc -eq 142 ]] && rc=124
    return $rc
  fi

  "$@"
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

BUILD_LOG="$LOG_DIR/build.log"
STAGE1_LOG="$LOG_DIR/stage1.log"
STAGE2_LOG="$LOG_DIR/stage2.log"
NO_RUST_MARKER_LOG="$LOG_DIR/no-rust-markers.log"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
rm -f "$BUILD_LOG" "$STAGE1_LOG" "$STAGE2_LOG" "$NO_RUST_MARKER_LOG"
rm -f "$SUMMARY_FILE" "$STAGE1_PATH" "$STAGE2_PATH"

echo "SELFHOST_CYCLE_GATE_START"
echo "stage1_path=$STAGE1_PATH"
echo "stage2_path=$STAGE2_PATH"
echo "cache_path=$CACHE_PATH"
echo "bootstrap_bundle_dir=$BOOTSTRAP_BUNDLE_DIR"
echo "bootstrap_state_base_dir=$BOOTSTRAP_STATE_BASE_DIR"
echo "seed_enforce=$SEED_ENFORCE"
echo "seed_path=$SEED_PATH"
echo "seed_trusted_key=$BOOTSTRAP_SEED_TRUSTED_KEY"
echo "cycle_force_dynamic=$CYCLE_FORCE_DYNAMIC"
echo "no_rust_marker_enforce=$NO_RUST_MARKER_ENFORCE"
echo "bootstrap_manifest=$BOOTSTRAP_MANIFEST_PATH"
echo "independence_contract=$INDEPENDENCE_CONTRACT_PATH"

if [ -f "$INDEPENDENCE_CONTRACT_PATH" ]; then
  # Validate schema via pure-Sounio kaxi-validate-evidence (replaces python json.loads).
  "$ROOT_DIR/../bin/kretikos" kaxi-validate-evidence "$INDEPENDENCE_CONTRACT_PATH" \
    --expect "schema=sounio.independence.contract.v1"
fi

if [ "$SKIP_BUILD" = "1" ]; then
  : >"$BUILD_LOG"
else
  {
    echo "SELFHOST_CYCLE_GATE_INFO build_step=disabled mode=repo-hard-no-rust"
    echo "hint=provide SOUC_BIN prebuilt compiler"
  } >"$BUILD_LOG"
fi

if [ ! -x "$SOUC_BIN" ]; then
  echo "error: missing compiler binary at $SOUC_BIN" >&2
  exit 1
fi

extract_cycle_digest() {
  local report_path="$1"
  # Pure-bash cycle digest extractor (replaces python json.load heredoc).
  # Reads stage1_digest, stage2_digest, deterministic, artifacts_checked
  # via kaxi-validate-evidence --print-or-empty. Asserts both digests
  # nonempty AND equal; emits 3 KEY=VALUE lines on success.
  local kretikos="$ROOT_DIR/../bin/kretikos"
  local stage1 stage2 det art
  stage1="$("$kretikos" kaxi-validate-evidence "$report_path" --print-or-empty "stage1_digest" 2>/dev/null || true)"
  stage2="$("$kretikos" kaxi-validate-evidence "$report_path" --print-or-empty "stage2_digest" 2>/dev/null || true)"
  det="$("$kretikos" kaxi-validate-evidence "$report_path" --print-or-empty "deterministic" 2>/dev/null || true)"
  art="$("$kretikos" kaxi-validate-evidence "$report_path" --print-or-empty "artifacts_checked" 2>/dev/null || true)"

  if [[ -z "$stage1" || -z "$stage2" ]]; then
    echo "missing cycle digest fields in $report_path" >&2
    return 1
  fi
  if [[ "$stage1" != "$stage2" ]]; then
    echo "non-deterministic cycle report $report_path stage1=$stage1 stage2=$stage2" >&2
    return 1
  fi
  # Match python's "1 if deterministic else 0" — JSON true -> "1", anything else -> "0".
  if [[ "$det" == "true" ]]; then
    det="1"
  else
    det="0"
  fi
  # Default missing artifacts_checked to 0 (matches python's payload.get("artifacts_checked", 0)).
  [[ -z "$art" ]] && art="0"

  echo "cycle_digest=$stage1"
  echo "deterministic=$det"
  echo "artifacts_checked=$art"
}

run_cycle_stage() {
  local stage_label="$1"
  local stage_log="$2"
  local stage_path="$3"
  local state_dir="$4"
  local cycle_report_path="$state_dir/bootstrap/cycle-report.v1.json"

  rm -rf "$state_dir"

  run_with_timeout "$COMPILE_TIMEOUT_SECS" env \
    SOUNIO_BOOTSTRAP_SEED_ENFORCE="$SEED_ENFORCE" \
    SOUNIO_BOOTSTRAP_SEED_PATH="$SEED_PATH" \
    SOUNIO_BOOTSTRAP_SEED_SHA256_PATH="$SEED_SHA256_PATH" \
    SOUNIO_BOOTSTRAP_SEED_SIG_PATH="$SEED_SIG_PATH" \
    SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY="$BOOTSTRAP_SEED_TRUSTED_KEY" \
    "$SOUC_BIN" bootstrap verify --bundle "$BOOTSTRAP_BUNDLE_DIR" >"$stage_log" 2>&1

  run_with_timeout "$COMPILE_TIMEOUT_SECS" env \
    SOUNIO_BOOTSTRAP_SEED_ENFORCE="$SEED_ENFORCE" \
    SOUNIO_BOOTSTRAP_SEED_PATH="$SEED_PATH" \
    SOUNIO_BOOTSTRAP_SEED_SHA256_PATH="$SEED_SHA256_PATH" \
    SOUNIO_BOOTSTRAP_SEED_SIG_PATH="$SEED_SIG_PATH" \
    SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY="$BOOTSTRAP_SEED_TRUSTED_KEY" \
    "$SOUC_BIN" bootstrap init --bundle "$BOOTSTRAP_BUNDLE_DIR" --state "$state_dir" >>"$stage_log" 2>&1

  run_with_timeout "$COMPILE_TIMEOUT_SECS" env \
    SOUNIO_BOOTSTRAP_SEED_ENFORCE="$SEED_ENFORCE" \
    SOUNIO_BOOTSTRAP_SEED_PATH="$SEED_PATH" \
    SOUNIO_BOOTSTRAP_SEED_SHA256_PATH="$SEED_SHA256_PATH" \
    SOUNIO_BOOTSTRAP_SEED_SIG_PATH="$SEED_SIG_PATH" \
    SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY="$BOOTSTRAP_SEED_TRUSTED_KEY" \
    "$SOUC_BIN" bootstrap cycle --state "$state_dir" >>"$stage_log" 2>&1

  if [ ! -f "$cycle_report_path" ]; then
    echo "error: missing ${stage_label} cycle report at $cycle_report_path" >&2
    exit 1
  fi

  extract_cycle_digest "$cycle_report_path" >"$stage_path"
  echo "SELFHOST_CYCLE_GATE_INFO stage=$stage_label artifact_source=cycle_report state_dir=$state_dir"
}

STAGE1_STATE_DIR="$BOOTSTRAP_STATE_BASE_DIR/stage1"
STAGE2_STATE_DIR="$BOOTSTRAP_STATE_BASE_DIR/stage2"

run_cycle_stage "stage1" "$STAGE1_LOG" "$STAGE1_PATH" "$STAGE1_STATE_DIR"
run_cycle_stage "stage2" "$STAGE2_LOG" "$STAGE2_PATH" "$STAGE2_STATE_DIR"

if ! cmp -s "$STAGE1_PATH" "$STAGE2_PATH"; then
  echo "error: self-host cycle mismatch: $STAGE1_PATH != $STAGE2_PATH" >&2
  exit 1
fi

if [ "$NO_RUST_MARKER_ENFORCE" = "1" ]; then
  if ! run_with_timeout "$NO_RUST_MARKER_TIMEOUT_SECS" \
    bash scripts/assert_no_rust_markers.sh \
    "$STAGE1_LOG" \
    "$STAGE2_LOG" >"$NO_RUST_MARKER_LOG" 2>&1; then
    echo "error: rust marker leakage detected (see $NO_RUST_MARKER_LOG)" >&2
    cat "$NO_RUST_MARKER_LOG" >&2
    exit 1
  fi
fi

SHA_STAGE1="$(sha256sum "$STAGE1_PATH" | awk '{print $1}')"
SHA_STAGE2="$(sha256sum "$STAGE2_PATH" | awk '{print $1}')"

{
  echo "stage1_sha256=$SHA_STAGE1"
  echo "stage2_sha256=$SHA_STAGE2"
  echo "stage_bytes=$(wc -c <"$STAGE1_PATH")"
  echo "stage1_state_dir=$STAGE1_STATE_DIR"
  echo "stage2_state_dir=$STAGE2_STATE_DIR"
} >"$SUMMARY_FILE"

echo "SELFHOST_CYCLE_GATE_DONE sha=$SHA_STAGE1 bytes=$(wc -c <"$STAGE1_PATH")"
