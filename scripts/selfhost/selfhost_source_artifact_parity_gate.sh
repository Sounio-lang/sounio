#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ACCEPTED_ARTIFACT_BIN="${ACCEPTED_ARTIFACT_BIN:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
SOURCE_FILE="${SOURCE_FILE:-self-hosted/compiler/lean_single.sio}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-source-artifact-parity-gate}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"

REBUILD_BIN="${REBUILD_BIN:-$ARTIFACT_DIR/rebuild-from-source.elf}"
HEAD_BIN="${HEAD_BIN:-$ARTIFACT_DIR/current-head.elf}"
SUITE_FILTER="${SUITE_FILTER:-}"

REPORT_JSON="$ARTIFACT_DIR/suite_delta.v1.json"
REPORT_MD="$ARTIFACT_DIR/suite_delta.v1.md"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"

echo "SELFHOST_SOURCE_ARTIFACT_PARITY_GATE_START"
echo "accepted_artifact_bin=$ACCEPTED_ARTIFACT_BIN"
echo "source_file=$SOURCE_FILE"
echo "work_dir=$WORK_DIR"
echo "rebuild_bin=$REBUILD_BIN"
echo "head_bin=$HEAD_BIN"
echo "suite_filter=$SUITE_FILTER"

if [ ! -x "$ACCEPTED_ARTIFACT_BIN" ]; then
  echo "error: missing accepted artifact compiler at $ACCEPTED_ARTIFACT_BIN" >&2
  exit 1
fi

rm -f "$REBUILD_BIN" "$HEAD_BIN"

"$ACCEPTED_ARTIFACT_BIN" "$SOURCE_FILE" "$REBUILD_BIN" >"$LOG_DIR/rebuild.compile.stdout" 2>"$LOG_DIR/rebuild.compile.stderr"
chmod +x "$REBUILD_BIN"
"$REBUILD_BIN" "$SOURCE_FILE" "$HEAD_BIN" >"$LOG_DIR/head.compile.stdout" 2>"$LOG_DIR/head.compile.stderr"
chmod +x "$HEAD_BIN"

run_suite() {
  local label="$1"
  local compiler_bin="$2"
  local log_path="$3"
  local exit_path="$4"
  local suite_exit=0

  set +e
  if [ -n "$SUITE_FILTER" ]; then
    env SOUNIO_TEST_NATIVE_BIN="$compiler_bin" bash "$ROOT_DIR/scripts/run_sio_test_suite.sh" "$SUITE_FILTER" >"$log_path" 2>&1
  else
    env SOUNIO_TEST_NATIVE_BIN="$compiler_bin" bash "$ROOT_DIR/scripts/run_sio_test_suite.sh" >"$log_path" 2>&1
  fi
  suite_exit=$?
  set -e
  echo "$suite_exit" >"$exit_path"
  echo "SELFHOST_SOURCE_ARTIFACT_PARITY_GATE_RUN label=$label compiler=$compiler_bin exit=$suite_exit log=$log_path"
}

run_suite "accepted_artifact_baseline" "$ACCEPTED_ARTIFACT_BIN" "$LOG_DIR/accepted.log" "$ARTIFACT_DIR/accepted.exit"
run_suite "rebuild_from_source_baseline" "$REBUILD_BIN" "$LOG_DIR/rebuild.log" "$ARTIFACT_DIR/rebuild.exit"
run_suite "current_head" "$HEAD_BIN" "$LOG_DIR/head.log" "$ARTIFACT_DIR/head.exit"

python3 "$ROOT_DIR/scripts/selfhost/selfhost_suite_delta.py" \
  --accepted-log "$LOG_DIR/accepted.log" \
  --rebuild-log "$LOG_DIR/rebuild.log" \
  --head-log "$LOG_DIR/head.log" \
  --json-out "$REPORT_JSON" \
  --md-out "$REPORT_MD"

{
  echo "accepted_artifact_bin=$ACCEPTED_ARTIFACT_BIN"
  echo "source_file=$SOURCE_FILE"
  echo "rebuild_bin=$REBUILD_BIN"
  echo "head_bin=$HEAD_BIN"
  echo "accepted_log=$LOG_DIR/accepted.log"
  echo "rebuild_log=$LOG_DIR/rebuild.log"
  echo "head_log=$LOG_DIR/head.log"
  echo "report_json=$REPORT_JSON"
  echo "report_md=$REPORT_MD"
} >"$SUMMARY_FILE"

echo "SELFHOST_SOURCE_ARTIFACT_PARITY_GATE_PASS report_json=$REPORT_JSON report_md=$REPORT_MD"
exit 0
