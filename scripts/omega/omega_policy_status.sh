#!/usr/bin/env bash
set -euo pipefail
# Resolved here, at the top, because this script changes directory later and a
# relative BASH_SOURCE stops resolving once it does.
_SOUC_GUARD_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/souc_verb_guard.sh"
. "$_SOUC_GUARD_LIB"

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
  --pin-digest <sha256>    Expected pinned payload digest (default: env/freeze artifact)
  --freeze <path>          Baseline freeze artifact path (default: artifacts/omega/baseline_freeze.v1.json)
EOF
}

POLICY_PATH=""
SOUC_BIN="souc"
CORPUS_PATH="benchmarks/independence"
CANONICAL_ENV_PATH="artifacts/omega/canonical_key.env"
SMOKE_ENV_PATH="artifacts/omega/policy_smoke.env"
SMOKE_OUT_PATH="artifacts/omega/policy_status_smoke.v2.json"
PIN_DIGEST="${OMEGA_POLICY_PIN_DIGEST:-}"
FREEZE_PATH="${OMEGA_POLICY_PIN_FREEZE_PATH:-${OMEGA_BASELINE_FREEZE_OUT:-artifacts/omega/baseline_freeze.v1.json}}"

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
    --pin-digest)
      PIN_DIGEST="${2:-}"
      shift 2
      ;;
    --freeze)
      FREEZE_PATH="${2:-}"
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
if [ ! -f "$SOUNIO_POLICY_VERIFY_KEY_PATH" ]; then
  echo "error: verification key path not found: $SOUNIO_POLICY_VERIFY_KEY_PATH" >&2
  exit 2
fi

ACTUAL_FINGERPRINT="$(python3 - "$SOUNIO_POLICY_VERIFY_KEY_PATH" <<'PY'
import hashlib
import sys
from pathlib import Path

pub_hex = Path(sys.argv[1]).read_text().strip()
print(hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest())
PY
)"
EXPECTED_FINGERPRINT="${OMEGA_CANONICAL_PUBKEY_FINGERPRINT:-}"
if [ -n "$EXPECTED_FINGERPRINT" ] && [ "$ACTUAL_FINGERPRINT" != "$EXPECTED_FINGERPRINT" ]; then
  echo "error: canonical verification key fingerprint mismatch expected=$EXPECTED_FINGERPRINT actual=$ACTUAL_FINGERPRINT" >&2
  exit 2
fi

set +e
STATUS_OUTPUT="$(
  # Refuse before the work, and name what is actually missing: the `opt`
  # verbs went with the Rust crate (79acc192e1) and the fall-through
  # diagnostic reports a missing FILE. See scripts/lib/souc_verb_guard.sh.
  require_souc_verb "$SOUC_BIN" opt "reading signed policy status"
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
if ! grep -q "opt policy canonical: signature=verified" <<<"$STATUS_OUTPUT"; then
  echo "error: canonical in-place signature attachment is not verified for $STATUS_POLICY_PATH" >&2
  exit 2
fi
if ! grep -q "fingerprint=$ACTUAL_FINGERPRINT" <<<"$STATUS_OUTPUT"; then
  echo "error: canonical fingerprint mismatch in status output for $STATUS_POLICY_PATH" >&2
  exit 2
fi

REQUIRE_PIN="${OMEGA_REQUIRE_PINNED_DIGEST:-0}"
if [ "$REQUIRE_PIN" = "1" ] && [ -z "$PIN_DIGEST" ] && [ -f "$FREEZE_PATH" ]; then
  PIN_DIGEST="$(python3 - "$FREEZE_PATH" <<'PY'
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
    echo "error: invalid pinned digest (expected 64 hex chars): $PIN_DIGEST" >&2
    exit 2
  fi
  PIN_DIGEST="${PIN_DIGEST,,}"
fi

if [ -n "$PIN_DIGEST" ]; then
  REQUIRE_PIN=1
fi
if [ "$REQUIRE_PIN" = "1" ]; then
  if ! grep -q "opt policy pinned: status=verified" <<<"$STATUS_OUTPUT"; then
    echo "error: pinned digest verification failed for $STATUS_POLICY_PATH" >&2
    exit 2
  fi
  if [ -n "$PIN_DIGEST" ] && ! grep -q "pinned=$PIN_DIGEST" <<<"$STATUS_OUTPUT"; then
    echo "error: status output pinned digest does not match expected freeze digest for $STATUS_POLICY_PATH" >&2
    exit 2
  fi
fi

echo "signature=verified (canonical in-place) fingerprint=$ACTUAL_FINGERPRINT"
