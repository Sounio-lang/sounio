#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${SOUNIO_DIAG_RUN_DIR:-}"
DIAG_ROOT="$ROOT_DIR/artifacts/diagnostic"
KEEP_RUNS="${SOUNIO_DIAG_KEEP_RUNS:-4}"

prune_old_runs() {
  local root="$1"
  local keep="$2"
  local run
  local -a runs

  [[ -d "$root" ]] || return 0
  if ! [[ "$keep" =~ ^[0-9]+$ ]] || (( keep < 1 )); then
    keep=4
  fi

  mapfile -t runs < <(find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | awk '{print $2}')
  if (( ${#runs[@]} <= keep )); then
    return 0
  fi

  for ((i = keep; i < ${#runs[@]}; i++)); do
    run="${runs[i]}"
    find "$run" -mindepth 1 -delete 2>/dev/null || true
    rmdir "$run" 2>/dev/null || true
  done
}

if [[ -z "$RUN_DIR" ]]; then
  mkdir -p "$DIAG_ROOT"
  prune_old_runs "$DIAG_ROOT" "$KEEP_RUNS"
  TS="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="$DIAG_ROOT/$TS"
fi

mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
mkdir -p "$RUN_DIR"/{cargo-home,cargo-target,npm-cache,logs}

DEFAULT_CARGO_HOME="$RUN_DIR/cargo-home"
DEFAULT_CARGO_TARGET_DIR="$RUN_DIR/cargo-target"
DEFAULT_NPM_CACHE="$RUN_DIR/npm-cache"

# Optional overrides for faster repeated diagnostics:
# - SOUNIO_DIAG_SHARED_CARGO_HOME=/path/to/shared/cargo-home
# - SOUNIO_DIAG_CARGO_HOME=/explicit/cargo-home
# - SOUNIO_DIAG_CARGO_TARGET_DIR=/explicit/target-dir
# - SOUNIO_DIAG_CARGO_NET_OFFLINE=1   (sets CARGO_NET_OFFLINE=true)
if [[ -n "${SOUNIO_DIAG_SHARED_CARGO_HOME:-}" ]]; then
  DEFAULT_CARGO_HOME="$SOUNIO_DIAG_SHARED_CARGO_HOME"
fi

EFFECTIVE_CARGO_HOME="${SOUNIO_DIAG_CARGO_HOME:-$DEFAULT_CARGO_HOME}"
EFFECTIVE_CARGO_TARGET_DIR="${SOUNIO_DIAG_CARGO_TARGET_DIR:-$DEFAULT_CARGO_TARGET_DIR}"
EFFECTIVE_NPM_CACHE="${SOUNIO_DIAG_NPM_CACHE:-$DEFAULT_NPM_CACHE}"

mkdir -p "$EFFECTIVE_CARGO_HOME" "$EFFECTIVE_CARGO_TARGET_DIR" "$EFFECTIVE_NPM_CACHE"

export SOUNIO_DIAG_RUN_DIR="$RUN_DIR"
export SOUNIO_DIAG_ARTIFACT_DIR="$RUN_DIR"
export SOUNIO_DIAG_ISOLATED=1
export CARGO_HOME="$EFFECTIVE_CARGO_HOME"
export CARGO_TARGET_DIR="$EFFECTIVE_CARGO_TARGET_DIR"
export CARGO_INCREMENTAL=0
export npm_config_cache="$EFFECTIVE_NPM_CACHE"
export NPM_CONFIG_CACHE="$EFFECTIVE_NPM_CACHE"

if [[ "${SOUNIO_DIAG_CARGO_NET_OFFLINE:-0}" == "1" ]]; then
  export CARGO_NET_OFFLINE=true
fi

if [[ $# -eq 0 ]]; then
  cat <<EOF
SOUNIO_DIAG_RUN_DIR=$SOUNIO_DIAG_RUN_DIR
SOUNIO_DIAG_ISOLATED=$SOUNIO_DIAG_ISOLATED
CARGO_HOME=$CARGO_HOME
CARGO_TARGET_DIR=$CARGO_TARGET_DIR
npm_config_cache=$npm_config_cache
CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-}
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
