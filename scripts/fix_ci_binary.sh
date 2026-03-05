#!/usr/bin/env bash
set -euo pipefail

# Resolve a reproducible standard souc binary for CI jobs.
# Default pin tracks the latest stable release asset with signatures.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${SOUNIO_CI_SOUC_VERSION:=0.100.3}"
: "${OMEGA_SOUC_REQUIRE_PINNED:=1}"
: "${OMEGA_SOUC_ALLOW_LOCAL_FALLBACK:=1}"

RESOLVER_SCRIPT="$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh"
if [[ ! -x "$RESOLVER_SCRIPT" ]]; then
  chmod +x "$RESOLVER_SCRIPT"
fi

SOUC_BIN="$({
  SOUNIO_SOUC_VERSION="$SOUNIO_CI_SOUC_VERSION" \
  SOUNIO_SOUC_VARIANT="std" \
  OMEGA_SOUC_REQUIRE_PINNED="$OMEGA_SOUC_REQUIRE_PINNED" \
  OMEGA_SOUC_ALLOW_LOCAL_FALLBACK="$OMEGA_SOUC_ALLOW_LOCAL_FALLBACK" \
    "$RESOLVER_SCRIPT" --print-path
} )"

if [[ ! -x "$SOUC_BIN" ]]; then
  echo "error: resolved souc binary is not executable: $SOUC_BIN" >&2
  exit 2
fi

SOUC_BIN="$(realpath "$SOUC_BIN")"
echo "Using resolved souc binary: $SOUC_BIN"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "SOUC_BIN=$SOUC_BIN"
    echo "SOUNIO_SOUC_VERSION=$SOUNIO_CI_SOUC_VERSION"
  } >> "$GITHUB_ENV"
else
  echo "SOUC_BIN=$SOUC_BIN"
  echo "SOUNIO_SOUC_VERSION=$SOUNIO_CI_SOUC_VERSION"
fi
