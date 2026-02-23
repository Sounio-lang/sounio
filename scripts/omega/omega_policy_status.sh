#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: omega_policy_status.sh --policy <path> [options]

Options:
  --policy <path>          Policy source path (required)
  --souc <path>            souc binary/command (default: souc)
  --corpus <path>          Corpus path for policy train signing (default: benchmarks/independence)
  --canonical-env <path>   Canonical key env file path (default: artifacts/omega/canonical_key.env)
  --smoke-env <path>       Policy smoke env file path (default: artifacts/omega/policy_smoke.env)
  --smoke-out <path>       Signed smoke policy output path (default: artifacts/omega/policy_status_smoke.v2.json)
EOF
}

POLICY_PATH=""
SOUC_BIN="souc"
CORPUS_PATH="benchmarks/independence"
CANONICAL_ENV_PATH="artifacts/omega/canonical_key.env"
SMOKE_ENV_PATH="artifacts/omega/policy_smoke.env"
SMOKE_OUT_PATH="artifacts/omega/policy_status_smoke.v2.json"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --policy)
      POLICY_PATH="${2:-}"
      shift 2
      ;;
    --souc)
      SOUC_BIN="${2:-}"
      shift 2
      ;;
    --corpus)
      CORPUS_PATH="${2:-}"
      shift 2
      ;;
    --canonical-env)
      CANONICAL_ENV_PATH="${2:-}"
      shift 2
      ;;
    --smoke-env)
      SMOKE_ENV_PATH="${2:-}"
      shift 2
      ;;
    --smoke-out)
      SMOKE_OUT_PATH="${2:-}"
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

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREP_SCRIPT="${OMEGA_POLICY_PREP_SCRIPT:-$SCRIPT_ROOT/omega/omega_prepare_policy_smoke.sh}"
if [ ! -x "$PREP_SCRIPT" ]; then
  echo "error: policy prepare script not executable: $PREP_SCRIPT" >&2
  exit 2
fi

"$PREP_SCRIPT" \
  --policy "$POLICY_PATH" \
  --souc "$SOUC_BIN" \
  --corpus "$CORPUS_PATH" \
  --out "$SMOKE_OUT_PATH" \
  --env-out "$SMOKE_ENV_PATH" \
  --canonical-env "$CANONICAL_ENV_PATH" >/dev/null

if [ ! -f "$SMOKE_ENV_PATH" ]; then
  echo "error: missing policy smoke env file: $SMOKE_ENV_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$SMOKE_ENV_PATH"

STATUS_POLICY_PATH="${SOUNIO_POLICY_STATUS_PATH:-$POLICY_PATH}"
if [ -z "${SOUNIO_POLICY_VERIFY_KEY_PATH:-}" ]; then
  echo "error: SOUNIO_POLICY_VERIFY_KEY_PATH is required for canonical verification" >&2
  exit 2
fi

set +e
STATUS_OUTPUT="$(
  SOUNIO_POLICY_VERIFY_KEY_PATH="$SOUNIO_POLICY_VERIFY_KEY_PATH" \
    "$SOUC_BIN" opt policy status --policy "$STATUS_POLICY_PATH" 2>&1
)"
STATUS_CODE=$?
set -e
echo "$STATUS_OUTPUT"
if [ $STATUS_CODE -ne 0 ]; then
  echo "error: canonical policy status command failed: policy=$STATUS_POLICY_PATH" >&2
  exit $STATUS_CODE
fi

if ! grep -q "signature=verified" <<<"$STATUS_OUTPUT"; then
  echo "error: canonical policy verification failed for $STATUS_POLICY_PATH" >&2
  exit 2
fi

echo "signature=verified (canonical bootstrap key) fingerprint=${OMEGA_CANONICAL_PUBKEY_FINGERPRINT:-unknown}"
