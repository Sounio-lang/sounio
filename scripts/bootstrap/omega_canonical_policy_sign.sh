#!/usr/bin/env bash
set -euo pipefail
# Resolved here, at the top, because this script changes directory later and a
# relative BASH_SOURCE stops resolving once it does.
_SOUC_GUARD_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/souc_verb_guard.sh"
. "$_SOUC_GUARD_LIB"

usage() {
  cat <<'EOF'
Usage: omega_canonical_policy_sign.sh --policy <path> [options]

Options:
  --policy <path>          Input optimization policy JSON (required)
  --out <path>             Output path (default: in-place overwrite)
  --souc <path>            souc binary/command (default: souc)
  --canonical-env <path>   Canonical key env path (default: artifacts/omega/canonical_key.env)
  --pin-digest <sha256>    Optional pinned payload digest (default: env/freeze artifact)
EOF
}

POLICY_PATH=""
OUT_PATH=""
SOUC_BIN="${SOUC_BIN:-souc}"
CANONICAL_ENV_PATH="${OMEGA_CANONICAL_ENV_OUT:-artifacts/omega/canonical_key.env}"
PIN_DIGEST="${OMEGA_POLICY_PIN_DIGEST:-}"
PIN_FREEZE_PATH="${OMEGA_POLICY_PIN_FREEZE_PATH:-${OMEGA_BASELINE_FREEZE_OUT:-artifacts/omega/baseline_freeze.v1.json}}"
PIN_FROM_FREEZE="${OMEGA_POLICY_PIN_FROM_FREEZE:-0}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --policy)
      POLICY_PATH="${2:-}"
      shift 2
      ;;
    --out)
      OUT_PATH="${2:-}"
      shift 2
      ;;
    --souc)
      SOUC_BIN="${2:-}"
      shift 2
      ;;
    --canonical-env)
      CANONICAL_ENV_PATH="${2:-}"
      shift 2
      ;;
    --pin-digest)
      PIN_DIGEST="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$POLICY_PATH" ]; then
  echo "error: --policy is required" >&2
  usage >&2
  exit 2
fi
if [ ! -f "$POLICY_PATH" ]; then
  echo "error: policy path not found: $POLICY_PATH" >&2
  exit 2
fi
if [ -z "$OUT_PATH" ]; then
  OUT_PATH="$POLICY_PATH"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Refuse before the work, and name what is actually missing: the `opt`
# verbs went with the Rust crate (79acc192e1) and the fall-through
# diagnostic reports a missing FILE. See scripts/lib/souc_verb_guard.sh.
require_souc_verb "$SOUC_BIN" opt "signing the canonical policy"

CANONICAL_BOOTSTRAP_SCRIPT="${OMEGA_CANONICAL_BOOTSTRAP_SCRIPT:-$SCRIPT_DIR/omega_canonical_key_bootstrap.sh}"
if [ ! -x "$CANONICAL_BOOTSTRAP_SCRIPT" ]; then
  echo "error: canonical bootstrap script not executable: $CANONICAL_BOOTSTRAP_SCRIPT" >&2
  exit 2
fi

if [ -f "$CANONICAL_ENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$CANONICAL_ENV_PATH"
fi
if [ ! -f "${OMEGA_CANONICAL_PRIVKEY:-}" ] || [ ! -f "${OMEGA_CANONICAL_PUBKEY:-}" ]; then
  "$CANONICAL_BOOTSTRAP_SCRIPT" --env-out "$CANONICAL_ENV_PATH" >/dev/null
  # shellcheck disable=SC1090
  source "$CANONICAL_ENV_PATH"
fi
if [ ! -f "$OMEGA_CANONICAL_PRIVKEY" ] || [ ! -f "$OMEGA_CANONICAL_PUBKEY" ]; then
  echo "error: canonical key bootstrap did not produce required key files" >&2
  exit 2
fi

if [ -z "$PIN_DIGEST" ] && [ "$PIN_FROM_FREEZE" = "1" ] && [ -f "$PIN_FREEZE_PATH" ]; then
  PIN_DIGEST="$(python3 - "$PIN_FREEZE_PATH" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
value = payload.get("policy_pinned_digest_sha256", "")
print(value if isinstance(value, str) else "")
PY
)"
fi

if [ -n "$PIN_DIGEST" ]; then
  if ! [[ "$PIN_DIGEST" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "error: invalid --pin-digest value (expected 64 hex chars): $PIN_DIGEST" >&2
    exit 2
  fi
  PIN_DIGEST="${PIN_DIGEST,,}"
fi

SIGN_ARGS=(opt policy sign --policy "$POLICY_PATH" --out "$OUT_PATH")
if [ -n "$PIN_DIGEST" ]; then
  SIGN_ARGS+=(--pin-digest "$PIN_DIGEST")
fi

OMEGA_CANONICAL_PRIVKEY="$OMEGA_CANONICAL_PRIVKEY" \
OMEGA_CANONICAL_PUBKEY="$OMEGA_CANONICAL_PUBKEY" \
  "$SOUC_BIN" "${SIGN_ARGS[@]}"
