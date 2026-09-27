#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BUNDLE_DIR="${BUNDLE_DIR:-bootstrap}"
PRIVATE_KEY="${PRIVATE_KEY:-}"
KEY_ID_OVERRIDE="${KEY_ID:-}"

usage() {
  cat <<USAGE
Usage: $0 [--bundle DIR] --private-key FILE [--key-id ID]

Signs bootstrap manifest.v2 and all declared artifacts with Ed25519.

Options:
  --bundle DIR         Bundle root (default: bootstrap)
  --private-key FILE   Ed25519 private key PEM (required)
  --key-id ID          Override manifest key_id when writing public key file
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      BUNDLE_DIR="$2"
      shift 2
      ;;
    --private-key)
      PRIVATE_KEY="$2"
      shift 2
      ;;
    --key-id)
      KEY_ID_OVERRIDE="$2"
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

if [[ -z "$PRIVATE_KEY" ]]; then
  echo "error: --private-key is required" >&2
  usage
  exit 1
fi

if [[ ! -f "$PRIVATE_KEY" ]]; then
  echo "error: private key not found: $PRIVATE_KEY" >&2
  exit 1
fi

MANIFEST_PATH=""
if [[ -f "$BUNDLE_DIR/artifacts/manifest.v2.json" ]]; then
  MANIFEST_PATH="$BUNDLE_DIR/artifacts/manifest.v2.json"
elif [[ -f "$BUNDLE_DIR/manifest.v2.json" ]]; then
  MANIFEST_PATH="$BUNDLE_DIR/manifest.v2.json"
else
  echo "error: manifest.v2.json not found under $BUNDLE_DIR" >&2
  exit 1
fi

KEY_ID="$(python3 - <<'PY' "$MANIFEST_PATH"
import json,sys
with open(sys.argv[1],"r",encoding="utf-8") as f:
    m=json.load(f)
print(m.get("key_id",""))
PY
)"
if [[ -n "$KEY_ID_OVERRIDE" ]]; then
  KEY_ID="$KEY_ID_OVERRIDE"
fi
if [[ -z "$KEY_ID" ]]; then
  echo "error: key_id missing in manifest and not provided via --key-id" >&2
  exit 1
fi

mkdir -p "$BUNDLE_DIR/artifacts/keys"
PUB_PATH="$BUNDLE_DIR/artifacts/keys/$KEY_ID.pub"
openssl pkey -in "$PRIVATE_KEY" -pubout -outform DER 2>/dev/null | tail -c 32 | xxd -p -c 256 > "$PUB_PATH"

mapfile -t ARTIFACT_PATHS < <(python3 - <<'PY' "$MANIFEST_PATH"
import json,sys
with open(sys.argv[1],"r",encoding="utf-8") as f:
    m=json.load(f)
for a in m.get("artifacts",[]):
    p=a.get("path","").strip()
    if p:
        print(p)
PY
)

TMP_SIG="$(mktemp)"
trap 'rm -f "$TMP_SIG"' EXIT

for rel in "${ARTIFACT_PATHS[@]}"; do
  src="$BUNDLE_DIR/$rel"
  if [[ ! -f "$src" ]]; then
    echo "error: artifact declared in manifest is missing: $src" >&2
    exit 1
  fi
  openssl pkeyutl -sign -inkey "$PRIVATE_KEY" -rawin -in "$src" -out "$TMP_SIG" >/dev/null 2>&1
  xxd -p -c 256 "$TMP_SIG" > "$src.sig"
  echo "signed artifact: $src.sig"
done

openssl pkeyutl -sign -inkey "$PRIVATE_KEY" -rawin -in "$MANIFEST_PATH" -out "$TMP_SIG" >/dev/null 2>&1
xxd -p -c 256 "$TMP_SIG" > "$MANIFEST_PATH.sig"
echo "signed manifest: $MANIFEST_PATH.sig"

echo "bootstrap signing complete (key_id=$KEY_ID)"
