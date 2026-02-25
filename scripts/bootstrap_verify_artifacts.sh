#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/target/release/souc}"
BUNDLE_DIR="${BUNDLE_DIR:-bootstrap}"

usage() {
  cat <<USAGE
Usage: $0 [--bundle DIR] [--souc PATH]

Verifies bootstrap manifest.v2 + artifact signatures using souc bootstrap verify.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      BUNDLE_DIR="$2"
      shift 2
      ;;
    --souc)
      SOUC_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "$SOUC_BIN" ]]; then
  echo "error: souc binary not found/executable: $SOUC_BIN" >&2
  exit 1
fi

"$SOUC_BIN" bootstrap verify --bundle "$BUNDLE_DIR"
