#!/usr/bin/env bash
set -euo pipefail
# Resolved here, at the top, because this script changes directory later and a
# relative BASH_SOURCE stops resolving once it does.
_SOUC_GUARD_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/souc_verb_guard.sh"
. "$_SOUC_GUARD_LIB"

usage() {
  cat <<'EOF'
Usage: omega_prepare_policy_smoke.sh --policy <path> [options]

Options:
  --policy <path>          Source optimization policy path (required)
  --souc <path>            souc binary/command (default: souc)
  --corpus <path>          Corpus path for policy train signing (default: benchmarks/independence)
  --out <path>             Signed policy output path (default: artifacts/omega/policy_status_smoke.v2.json)
  --env-out <path>         Env export file path (default: artifacts/omega/policy_smoke.env)
  --canonical-env <path>   Canonical key env path (default: artifacts/omega/canonical_key.env)
EOF
}

POLICY_PATH=""
SOUC_BIN="souc"
CORPUS_PATH="benchmarks/independence"
OUT_PATH="artifacts/omega/policy_status_smoke.v2.json"
ENV_OUT_PATH="artifacts/omega/policy_smoke.env"
CANONICAL_ENV_PATH="artifacts/omega/canonical_key.env"

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
    --out)
      OUT_PATH="${2:-}"
      shift 2
      ;;
    --env-out)
      ENV_OUT_PATH="${2:-}"
      shift 2
      ;;
    --canonical-env)
      CANONICAL_ENV_PATH="${2:-}"
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
if [ ! -e "$CORPUS_PATH" ]; then
  echo "error: corpus path not found: $CORPUS_PATH" >&2
  exit 2
fi
if ! command -v "$SOUC_BIN" >/dev/null 2>&1 && [ ! -x "$SOUC_BIN" ]; then
  echo "error: souc command not found: $SOUC_BIN" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANONICAL_BOOTSTRAP_SCRIPT="${OMEGA_CANONICAL_BOOTSTRAP_SCRIPT:-$SCRIPT_DIR/bootstrap/omega_canonical_key_bootstrap.sh}"
CANONICAL_POLICY_SIGN_SCRIPT="${OMEGA_CANONICAL_POLICY_SIGN_SCRIPT:-$SCRIPT_DIR/bootstrap/omega_canonical_policy_sign.sh}"
if [ ! -x "$CANONICAL_BOOTSTRAP_SCRIPT" ]; then
  echo "error: canonical bootstrap script not executable: $CANONICAL_BOOTSTRAP_SCRIPT" >&2
  exit 2
fi
if [ ! -x "$CANONICAL_POLICY_SIGN_SCRIPT" ]; then
  echo "error: canonical policy sign script not executable: $CANONICAL_POLICY_SIGN_SCRIPT" >&2
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

extract_field() {
  local policy="$1"
  python3 - "$policy" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text())
print(
    obj.get("schema", ""),
    obj.get("policy_id", ""),
    obj.get("policy_version", ""),
    obj.get("policy_mode", ""),
    sep="\t",
)
PY
}

IFS=$'\t' read -r POLICY_SCHEMA POLICY_ID POLICY_VERSION POLICY_MODE <<<"$(extract_field "$POLICY_PATH")"

mkdir -p "$(dirname "$ENV_OUT_PATH")"
mkdir -p "$(dirname "$OUT_PATH")"

if [ "$POLICY_SCHEMA" != "sounio.optimization.policy.v2" ]; then
  cat >"$ENV_OUT_PATH" <<EOF
export SOUNIO_POLICY_STATUS_PATH="$POLICY_PATH"
export SOUNIO_POLICY_SIGNING_KEY_PATH="$OMEGA_CANONICAL_PRIVKEY"
export SOUNIO_POLICY_VERIFY_KEY_PATH="$OMEGA_CANONICAL_PUBKEY"
export OMEGA_CANONICAL_PUBKEY_FINGERPRINT="$OMEGA_CANONICAL_PUBKEY_FINGERPRINT"
export OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP="$OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP"
EOF
  echo "omega_prepare_policy_smoke: schema=$POLICY_SCHEMA -> passthrough policy=$POLICY_PATH"
  exit 0
fi

if [ -z "$POLICY_ID" ] || [ -z "$POLICY_VERSION" ] || [ -z "$POLICY_MODE" ]; then
  echo "error: policy metadata missing required fields policy_id/policy_version/policy_mode" >&2
  exit 2
fi

META_PATH="${OUT_PATH}.meta.json"
SOURCE_SHA="$(sha256sum "$POLICY_PATH" | awk '{print $1}')"
SKIP_RETRAIN=0
if [ -f "$OUT_PATH" ] && [ -f "$META_PATH" ]; then
  if python3 - "$META_PATH" "$SOURCE_SHA" "$OMEGA_CANONICAL_PUBKEY_FINGERPRINT" "$POLICY_ID" "$POLICY_VERSION" "$POLICY_MODE" <<'PY'
import json
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "source_sha256": sys.argv[2],
    "canonical_pubkey_fingerprint": sys.argv[3],
    "policy_id": sys.argv[4],
    "policy_version": sys.argv[5],
    "policy_mode": sys.argv[6],
}
for key, value in expected.items():
    if str(meta.get(key, "")) != value:
        raise SystemExit(1)
PY
  then
    SKIP_RETRAIN=1
  fi
fi

if [ "$SKIP_RETRAIN" = "0" ]; then
  # Refuse before the work, and name what is actually missing: the `opt`
  # verbs went with the Rust crate (79acc192e1) and the fall-through
  # diagnostic reports a missing FILE. See scripts/lib/souc_verb_guard.sh.
  require_souc_verb "$SOUC_BIN" opt "training and signing a smoke policy"
  SOUNIO_POLICY_SIGNING_KEY_PATH="$OMEGA_CANONICAL_PRIVKEY" \
  SOUNIO_POLICY_VERIFY_KEY_PATH="$OMEGA_CANONICAL_PUBKEY" \
    "$SOUC_BIN" opt policy train \
    --corpus "$CORPUS_PATH" \
    --output "$OUT_PATH" \
    --policy-id "$POLICY_ID" \
    --policy-version "$POLICY_VERSION" \
    --mode "$POLICY_MODE" >/dev/null

  python3 - "$META_PATH" "$SOURCE_SHA" "$OMEGA_CANONICAL_PUBKEY_FINGERPRINT" "$POLICY_ID" "$POLICY_VERSION" "$POLICY_MODE" "$OUT_PATH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "schema": "sounio.omega.policy-smoke-meta.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_sha256": sys.argv[2],
    "canonical_pubkey_fingerprint": sys.argv[3],
    "policy_id": sys.argv[4],
    "policy_version": sys.argv[5],
    "policy_mode": sys.argv[6],
    "status_policy_path": sys.argv[7],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2))
PY
fi

"$CANONICAL_POLICY_SIGN_SCRIPT" \
  --policy "$OUT_PATH" \
  --out "$OUT_PATH" \
  --souc "$SOUC_BIN" \
  --canonical-env "$CANONICAL_ENV_PATH" >/dev/null

cat >"$ENV_OUT_PATH" <<EOF
export SOUNIO_POLICY_STATUS_PATH="$OUT_PATH"
export SOUNIO_POLICY_SIGNING_KEY_PATH="$OMEGA_CANONICAL_PRIVKEY"
export SOUNIO_POLICY_VERIFY_KEY_PATH="$OMEGA_CANONICAL_PUBKEY"
export OMEGA_CANONICAL_PUBKEY_FINGERPRINT="$OMEGA_CANONICAL_PUBKEY_FINGERPRINT"
export OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP="$OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP"
EOF

echo "omega_prepare_policy_smoke: source=$POLICY_PATH signed_policy=$OUT_PATH verify_key=$OMEGA_CANONICAL_PUBKEY"
