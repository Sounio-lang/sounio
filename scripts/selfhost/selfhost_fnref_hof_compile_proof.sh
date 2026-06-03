#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Focused proof for the named-function-reference/HOF cases that macOS arm64
# currently attests as known blockers. Linux must continue compiling them.
export MANIFEST_PATH="${MANIFEST_PATH:-tests/selfhost/aarch64_compile/manifest.tsv}"
export TARGET="${TARGET:-x86_64-linux}"
export WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-fnref-hof-compile-proof}"

echo "SELFHOST_FNREF_HOF_COMPILE_PROOF_START"
echo "manifest=$MANIFEST_PATH"
echo "target=$TARGET"
echo "work_dir=$WORK_DIR"

bash "$ROOT_DIR/scripts/selfhost/selfhost_aarch64_compile_proof.sh"

RESULTS_FILE="$WORK_DIR/artifacts/results.tsv"
if [ ! -f "$RESULTS_FILE" ]; then
  echo "error: missing compile proof results at $RESULTS_FILE" >&2
  exit 1
fi

if grep -Eq $'\t(macos_arm64_known_blocker|filtered|compile_fail|missing_output|empty_output|wrong_artifact_kind|missing_program)$' "$RESULTS_FILE"; then
  echo "error: fn-ref/HOF compile proof contains skipped or failed cases" >&2
  cat "$RESULTS_FILE" >&2
  exit 1
fi

echo "SELFHOST_FNREF_HOF_COMPILE_PROOF_DONE"
echo "results_file=$RESULTS_FILE"
