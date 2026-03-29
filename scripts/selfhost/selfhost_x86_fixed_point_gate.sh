#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
SOURCE_FILE="${SOURCE_FILE:-self-hosted/compiler/lean_single.sio}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-x86-fixed-point-gate}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"

GEN1_PATH="${GEN1_PATH:-$ARTIFACT_DIR/gen1.elf}"
GEN2_PATH="${GEN2_PATH:-$ARTIFACT_DIR/gen2.elf}"
GEN3_PATH="${GEN3_PATH:-$ARTIFACT_DIR/gen3.elf}"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
MD5_FILE="$ARTIFACT_DIR/md5.txt"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"

echo "SELFHOST_X86_FIXED_POINT_GATE_START"
echo "souc_native=$SOUC_NATIVE"
echo "source_file=$SOURCE_FILE"
echo "work_dir=$WORK_DIR"
echo "gen1_path=$GEN1_PATH"
echo "gen2_path=$GEN2_PATH"
echo "gen3_path=$GEN3_PATH"

if [ ! -x "$SOUC_NATIVE" ]; then
  echo "error: missing self-hosted native compiler at $SOUC_NATIVE" >&2
  exit 1
fi

rm -f "$GEN1_PATH" "$GEN2_PATH" "$GEN3_PATH" "$MD5_FILE"

"$SOUC_NATIVE" "$SOURCE_FILE" "$GEN1_PATH" >"$LOG_DIR/gen1.stdout" 2>"$LOG_DIR/gen1.stderr"
chmod +x "$GEN1_PATH"
"$GEN1_PATH" "$SOURCE_FILE" "$GEN2_PATH" >"$LOG_DIR/gen2.stdout" 2>"$LOG_DIR/gen2.stderr"
chmod +x "$GEN2_PATH"
"$GEN2_PATH" "$SOURCE_FILE" "$GEN3_PATH" >"$LOG_DIR/gen3.stdout" 2>"$LOG_DIR/gen3.stderr"
md5sum "$GEN2_PATH" "$GEN3_PATH" | tee "$MD5_FILE"

GEN2_MD5="$(awk 'NR==1 {print $1}' "$MD5_FILE")"
GEN3_MD5="$(awk 'NR==2 {print $1}' "$MD5_FILE")"

{
  echo "souc_native=$SOUC_NATIVE"
  echo "source_file=$SOURCE_FILE"
  echo "gen1_path=$GEN1_PATH"
  echo "gen2_path=$GEN2_PATH"
  echo "gen3_path=$GEN3_PATH"
  echo "gen2_md5=$GEN2_MD5"
  echo "gen3_md5=$GEN3_MD5"
  echo "md5_file=$MD5_FILE"
  echo "log_dir=$LOG_DIR"
} >"$SUMMARY_FILE"

if [ "$GEN2_MD5" != "$GEN3_MD5" ]; then
  echo "SELFHOST_X86_FIXED_POINT_GATE_FAIL gen2_md5=$GEN2_MD5 gen3_md5=$GEN3_MD5"
  exit 1
fi

echo "SELFHOST_X86_FIXED_POINT_GATE_PASS gen2_md5=$GEN2_MD5 gen3_md5=$GEN3_MD5"
exit 0
