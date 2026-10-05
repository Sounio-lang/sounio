#!/usr/bin/env bash
set -euo pipefail
# Resolved here, at the top, because this script changes directory later and a
# relative BASH_SOURCE stops resolving once it does.
_SOUC_GUARD_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/souc_verb_guard.sh"
. "$_SOUC_GUARD_LIB"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

# Refuse before the work, and name what is actually missing: the `bootstrap`
# verbs went with the Rust crate (79acc192e1) and the fall-through
# diagnostic reports a missing FILE. See scripts/lib/souc_verb_guard.sh.
require_souc_verb "$SOUC_BIN" bootstrap "verifying manifest.v2 + artifact signatures"
"$SOUC_BIN" bootstrap verify --bundle "$BUNDLE_DIR"
