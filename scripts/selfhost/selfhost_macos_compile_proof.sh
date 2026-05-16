#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
MANIFEST_PATH="${MANIFEST_PATH:-tests/selfhost/aarch64_compile/manifest.tsv}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-macos-compile-proof}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"
TIMEOUT_SECS="${TIMEOUT_SECS:-30}"
FILTER="${FILTER:-}"
TARGETS="${TARGETS:-aarch64-macos x86_64-macos}"

PASS_COUNT=0
FAIL_COUNT=0

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"

SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
RESULTS_FILE="$ARTIFACT_DIR/results.tsv"

cat >"$RESULTS_FILE" <<'EOF'
target	status	log_dir	artifact_dir
EOF

echo "SELFHOST_MACOS_COMPILE_PROOF_START"
echo "souc_native=$SOUC_NATIVE"
echo "manifest=$MANIFEST_PATH"
echo "work_dir=$WORK_DIR"
echo "targets=$TARGETS"

for target in $TARGETS; do
  target_work_dir="$WORK_DIR/$target"
  target_log="$LOG_DIR/$target.log"

  set +e
  SOUC_NATIVE="$SOUC_NATIVE" \
    MANIFEST_PATH="$MANIFEST_PATH" \
    WORK_DIR="$target_work_dir" \
    TIMEOUT_SECS="$TIMEOUT_SECS" \
    FILTER="$FILTER" \
    TARGET="$target" \
    bash "$ROOT_DIR/scripts/selfhost/selfhost_aarch64_compile_proof.sh" \
    >"$target_log" 2>&1
  target_rc=$?
  set -e

  if [ "$target_rc" -eq 0 ]; then
    PASS_COUNT=$((PASS_COUNT + 1))
    printf '%s\tok\t%s\t%s\n' "$target" "$target_log" "$target_work_dir/artifacts" >>"$RESULTS_FILE"
    echo "PASS [$target] compile proof ok"
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf '%s\tfail\t%s\t%s\n' "$target" "$target_log" "$target_work_dir/artifacts" >>"$RESULTS_FILE"
    echo "FAIL [$target] compile proof failed (see $target_log)"
    sed -n '1,120p' "$target_log" || true
  fi
done

{
  echo "summary_pass=$PASS_COUNT"
  echo "summary_fail=$FAIL_COUNT"
  echo "targets=$TARGETS"
  echo "manifest=$MANIFEST_PATH"
  echo "souc_native=$SOUC_NATIVE"
  echo "results_file=$RESULTS_FILE"
  echo "artifact_dir=$ARTIFACT_DIR"
  echo "log_dir=$LOG_DIR"
} >"$SUMMARY_FILE"

echo "SELFHOST_MACOS_COMPILE_PROOF_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT targets=$TARGETS"
echo "results_file=$RESULTS_FILE"

if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi

exit 0
