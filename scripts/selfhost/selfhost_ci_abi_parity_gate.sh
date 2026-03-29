#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BOOTSTRAP_BIN="${BOOTSTRAP_BIN:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
SOURCE_FILE="${SOURCE_FILE:-self-hosted/compiler/lean_single.sio}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-ci-abi-parity-gate}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"
ABI_WORK_DIR="$WORK_DIR/abi"
AARCH64_WORK_DIR="$WORK_DIR/aarch64"
HEAD_BIN="$ARTIFACT_DIR/current-head.elf"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
SUMMARY_JSON="$ARTIFACT_DIR/summary.v1.json"
SUMMARY_MD="$ARTIFACT_DIR/summary.v1.md"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"

echo "SELFHOST_CI_ABI_PARITY_GATE_START"
echo "bootstrap_bin=$BOOTSTRAP_BIN"
echo "source_file=$SOURCE_FILE"
echo "work_dir=$WORK_DIR"

if [ ! -x "$BOOTSTRAP_BIN" ]; then
  echo "error: missing bootstrap compiler at $BOOTSTRAP_BIN" >&2
  exit 1
fi

rm -f "$HEAD_BIN"
"$BOOTSTRAP_BIN" "$SOURCE_FILE" "$HEAD_BIN" >"$LOG_DIR/head.compile.stdout" 2>"$LOG_DIR/head.compile.stderr"
chmod +x "$HEAD_BIN"

env SOUC_NATIVE="$HEAD_BIN" WORK_DIR="$ABI_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_abi_parity_regression_gate.sh" >"$LOG_DIR/abi.stdout" 2>"$LOG_DIR/abi.stderr" || {
    echo "error: ABI/parity regression gate failed" >&2
    cat "$LOG_DIR/abi.stdout" >&2 || true
    cat "$LOG_DIR/abi.stderr" >&2 || true
    exit 1
  }

env SOUC_NATIVE="$HEAD_BIN" WORK_DIR="$AARCH64_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_aarch64_compile_proof.sh" >"$LOG_DIR/aarch64.stdout" 2>"$LOG_DIR/aarch64.stderr" || {
    echo "error: aarch64 compile-proof gate failed" >&2
    cat "$LOG_DIR/aarch64.stdout" >&2 || true
    cat "$LOG_DIR/aarch64.stderr" >&2 || true
    exit 1
  }

ABI_SUMMARY="$ABI_WORK_DIR/artifacts/summary.txt"
AARCH64_SUMMARY="$AARCH64_WORK_DIR/artifacts/summary.txt"
ABI_PASS="$(sed -n 's/^summary_pass=//p' "$ABI_SUMMARY" | head -n1)"
ABI_FAIL="$(sed -n 's/^summary_fail=//p' "$ABI_SUMMARY" | head -n1)"
ABI_SKIP="$(sed -n 's/^summary_skip=//p' "$ABI_SUMMARY" | head -n1)"
AARCH64_PASS="$(sed -n 's/^summary_pass=//p' "$AARCH64_SUMMARY" | head -n1)"
AARCH64_FAIL="$(sed -n 's/^summary_fail=//p' "$AARCH64_SUMMARY" | head -n1)"
AARCH64_SKIP="$(sed -n 's/^summary_skip=//p' "$AARCH64_SUMMARY" | head -n1)"

{
  echo "bootstrap_bin=$BOOTSTRAP_BIN"
  echo "source_file=$SOURCE_FILE"
  echo "head_bin=$HEAD_BIN"
  echo "abi_summary=$ABI_SUMMARY"
  echo "aarch64_summary=$AARCH64_SUMMARY"
  echo "abi_pass=$ABI_PASS"
  echo "abi_fail=$ABI_FAIL"
  echo "abi_skip=$ABI_SKIP"
  echo "aarch64_pass=$AARCH64_PASS"
  echo "aarch64_fail=$AARCH64_FAIL"
  echo "aarch64_skip=$AARCH64_SKIP"
} >"$SUMMARY_FILE"

cat >"$SUMMARY_JSON" <<EOF
{
  "schema": "sounio.selfhost_ci_abi_parity_gate.v1",
  "bootstrap_bin": "$BOOTSTRAP_BIN",
  "source_file": "$SOURCE_FILE",
  "head_bin": "$HEAD_BIN",
  "abi_gate": {
    "pass": $ABI_PASS,
    "fail": $ABI_FAIL,
    "skip": $ABI_SKIP
  },
  "aarch64_compile_proof": {
    "pass": $AARCH64_PASS,
    "fail": $AARCH64_FAIL,
    "skip": $AARCH64_SKIP
  }
}
EOF

cat >"$SUMMARY_MD" <<EOF
# Selfhost ABI/Parity CI Gate

- bootstrap_bin: $BOOTSTRAP_BIN
- current_head_bin: $HEAD_BIN
- abi_gate: pass=$ABI_PASS fail=$ABI_FAIL skip=$ABI_SKIP
- aarch64_compile_proof: pass=$AARCH64_PASS fail=$AARCH64_FAIL skip=$AARCH64_SKIP
EOF

echo "SELFHOST_CI_ABI_PARITY_GATE_DONE"
echo "summary_file=$SUMMARY_FILE"
exit 0
