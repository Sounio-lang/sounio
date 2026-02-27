#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${LSP_SMOKE_LOG_PATH:-$ROOT_DIR/artifacts/omega/lsp_smoke.log}"

mkdir -p "$(dirname "$LOG_PATH")"
: >"$LOG_PATH"

resolve_souc() {
  if [ -n "${SOUC_BIN:-}" ] && [ -x "${SOUC_BIN:-}" ]; then
    printf '%s\n' "$SOUC_BIN"
    return 0
  fi
  OMEGA_SOUC_REQUIRE_PINNED="${OMEGA_SOUC_REQUIRE_PINNED:-1}" \
  OMEGA_SOUC_ALLOW_LOCAL_FALLBACK="${OMEGA_SOUC_ALLOW_LOCAL_FALLBACK:-0}" \
    bash "$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh" --print-path
}

SOUC_RESOLVED="$(resolve_souc)"
if [ -z "$SOUC_RESOLVED" ] || [ ! -x "$SOUC_RESOLVED" ]; then
  echo "error: unable to resolve executable SOUC_BIN for LSP smoke gate" >&2
  exit 1
fi

export SOUNIO_REPO_HARD_NO_RUST="${SOUNIO_REPO_HARD_NO_RUST:-1}"
export SOUNIO_LSP_STRICT_NO_RUST="${SOUNIO_LSP_STRICT_NO_RUST:-1}"
export SOUNIO_LSP_SOUC_BIN="$SOUC_RESOLVED"

echo "LSP_SMOKE_GATE_START souc=$SOUC_RESOLVED strict_no_rust=$SOUNIO_LSP_STRICT_NO_RUST" | tee -a "$LOG_PATH"
bash "$ROOT_DIR/tools/lsp/test_smoke.sh" 2>&1 | tee -a "$LOG_PATH"
echo "LSP_SMOKE_PASS" | tee -a "$LOG_PATH"
