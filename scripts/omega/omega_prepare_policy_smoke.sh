#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: omega_prepare_policy_smoke.sh --policy <path> [options]

Options:
  --policy <path>      Source optimization policy path (required)
  --souc <path>        souc binary/command (default: souc)
  --corpus <path>      Corpus path for policy train signing (default: benchmarks/independence)
  --out <path>         Signed policy output path (default: artifacts/omega/policy_status_smoke.v2.json)
  --env-out <path>     Env export file path (default: artifacts/omega/policy_smoke.env)
  --key-dir <path>     Local key directory for policy smoke signing (default: artifacts/omega/policy_keys)
EOF
}

POLICY_PATH=""
SOUC_BIN="souc"
CORPUS_PATH="benchmarks/independence"
OUT_PATH="artifacts/omega/policy_status_smoke.v2.json"
ENV_OUT_PATH="artifacts/omega/policy_smoke.env"
KEY_DIR="artifacts/omega/policy_keys"

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
    --key-dir)
      KEY_DIR="${2:-}"
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
if ! command -v openssl >/dev/null 2>&1; then
  echo "error: openssl is required for local policy smoke key generation" >&2
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
EOF
  echo "omega_prepare_policy_smoke: schema=$POLICY_SCHEMA -> passthrough policy=$POLICY_PATH"
  exit 0
fi

if [ -z "$POLICY_ID" ] || [ -z "$POLICY_VERSION" ] || [ -z "$POLICY_MODE" ]; then
  echo "error: policy metadata missing required fields policy_id/policy_version/policy_mode" >&2
  exit 2
fi

PRIVATE_HEX_PATH="$KEY_DIR/local_ed25519_private.hex"
PUBLIC_HEX_PATH="$KEY_DIR/local_ed25519_public.hex"

ensure_policy_keypair() {
  mkdir -p "$KEY_DIR"
  if [ -s "$PRIVATE_HEX_PATH" ] && [ -s "$PUBLIC_HEX_PATH" ]; then
    return 0
  fi

  local pem_path="$KEY_DIR/local_ed25519_private.pem"
  local key_text_path="$KEY_DIR/local_ed25519_key.txt"
  openssl genpkey -algorithm ED25519 -out "$pem_path" >/dev/null 2>&1
  openssl pkey -in "$pem_path" -text -noout >"$key_text_path"

  python3 - "$key_text_path" "$PRIVATE_HEX_PATH" "$PUBLIC_HEX_PATH" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text().splitlines()
priv_out = Path(sys.argv[2])
pub_out = Path(sys.argv[3])

mode = None
data = {"priv": [], "pub": []}
for line in src:
    chunk = line.strip().lower().rstrip(":")
    if chunk == "priv":
        mode = "priv"
        continue
    if chunk == "pub":
        mode = "pub"
        continue
    if mode is None or not chunk:
        continue
    if not re.fullmatch(r"[0-9a-f:]+", chunk):
        continue
    token = chunk.replace(":", "")
    if token:
        data[mode].append(token)

priv = "".join(data["priv"])
pub = "".join(data["pub"])
if len(priv) != 64 or len(pub) != 64:
    raise SystemExit(
        f"invalid generated Ed25519 key lengths: priv={len(priv)} pub={len(pub)}"
    )
priv_out.write_text(priv + "\n")
pub_out.write_text(pub + "\n")
PY

  chmod 600 "$PRIVATE_HEX_PATH"
}

ensure_policy_keypair

SOUNIO_POLICY_SIGNING_KEY_PATH="$PRIVATE_HEX_PATH" \
SOUNIO_POLICY_VERIFY_KEY_PATH="$PUBLIC_HEX_PATH" \
  "$SOUC_BIN" opt policy train \
  --corpus "$CORPUS_PATH" \
  --output "$OUT_PATH" \
  --policy-id "$POLICY_ID" \
  --policy-version "$POLICY_VERSION" \
  --mode "$POLICY_MODE" >/dev/null

SOUNIO_POLICY_VERIFY_KEY_PATH="$PUBLIC_HEX_PATH" \
  "$SOUC_BIN" opt policy eval --policy "$OUT_PATH" >/dev/null

cat >"$ENV_OUT_PATH" <<EOF
export SOUNIO_POLICY_STATUS_PATH="$OUT_PATH"
export SOUNIO_POLICY_SIGNING_KEY_PATH="$PRIVATE_HEX_PATH"
export SOUNIO_POLICY_VERIFY_KEY_PATH="$PUBLIC_HEX_PATH"
EOF

echo "omega_prepare_policy_smoke: source=$POLICY_PATH signed_policy=$OUT_PATH verify_key=$PUBLIC_HEX_PATH"
