#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${SOUNIO_DIAG_RUN_DIR:-}"

if [[ -z "$RUN_DIR" ]]; then
  TS="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="$ROOT_DIR/artifacts/diagnostic/$TS"
fi

mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
mkdir -p "$RUN_DIR"/{cargo-home,cargo-target,npm-cache,logs}

export SOUNIO_DIAG_RUN_DIR="$RUN_DIR"
export SOUNIO_DIAG_ARTIFACT_DIR="$RUN_DIR"
export CARGO_HOME="$RUN_DIR/cargo-home"
export CARGO_TARGET_DIR="$RUN_DIR/cargo-target"
export CARGO_INCREMENTAL=0
export npm_config_cache="$RUN_DIR/npm-cache"
export NPM_CONFIG_CACHE="$RUN_DIR/npm-cache"

if [[ $# -eq 0 ]]; then
  cat <<EOF
SOUNIO_DIAG_RUN_DIR=$SOUNIO_DIAG_RUN_DIR
CARGO_HOME=$CARGO_HOME
CARGO_TARGET_DIR=$CARGO_TARGET_DIR
npm_config_cache=$npm_config_cache
EOF
  exit 0
fi

LABEL="${SOUNIO_DIAG_LOG_LABEL:-run}"
LOG_FILE="$RUN_DIR/logs/${LABEL}.log"

{
  echo "[diag] run_dir=$RUN_DIR"
  echo "[diag] command=$*"
  "$@"
} 2>&1 | tee "$LOG_FILE"
