#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: omega_canonical_key_bootstrap.sh [options]

Options:
  --env-out <path>   Path to write export statements
                     (default: artifacts/omega/canonical_key.env)
  --key-base <path>  Canonical key basename
                     (default: <repo>/keys/bootstrap_ed25519)
EOF
}

OMEGA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY_BASE="${OMEGA_CANONICAL_KEY_BASE:-$OMEGA_ROOT/keys/bootstrap_ed25519}"
ENV_OUT="${OMEGA_CANONICAL_ENV_OUT:-$OMEGA_ROOT/artifacts/omega/canonical_key.env}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --env-out)
      ENV_OUT="${2:-}"
      shift 2
      ;;
    --key-base)
      KEY_BASE="${2:-}"
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

PRIVATE_PATH="$KEY_BASE"
PUBLIC_PATH="${KEY_BASE}.pub"
TIMESTAMP_PATH="${KEY_BASE}.created_at"
COMMENT_PATH="${KEY_BASE}.comment"
KEY_COMMENT="Omega Canonical Bootstrap v1 – DO NOT DELETE"

is_hex_len() {
  local value="$1"
  local expected_len="$2"
  [[ "$value" =~ ^[0-9a-fA-F]+$ ]] || return 1
  [ "${#value}" -eq "$expected_len" ]
}

read_key_hex() {
  local path="$1"
  tr -d '[:space:]' <"$path"
}

write_exports() {
  local fingerprint="$1"
  local ts="$2"
  mkdir -p "$(dirname "$ENV_OUT")"
  cat >"$ENV_OUT" <<EOF
export OMEGA_CANONICAL_PRIVKEY="$PRIVATE_PATH"
export OMEGA_CANONICAL_PUBKEY="$PUBLIC_PATH"
export OMEGA_CANONICAL_PUBKEY_FINGERPRINT="$fingerprint"
export OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP="$ts"
export OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP_PATH="$TIMESTAMP_PATH"
export OMEGA_CANONICAL_KEY_COMMENT_PATH="$COMMENT_PATH"
EOF
  chmod 600 "$ENV_OUT" 2>/dev/null || true
}

compute_pubkey_fingerprint() {
  python3 - "$PUBLIC_PATH" <<'PY'
import hashlib
import sys
from pathlib import Path

pub_hex = Path(sys.argv[1]).read_text().strip()
pub = bytes.fromhex(pub_hex)
print(hashlib.sha256(pub).hexdigest())
PY
}

generate_with_openssl() {
  local tmpdir
  tmpdir="$(mktemp -d)"
  local pem="$tmpdir/canonical_key.pem"
  local key_text="$tmpdir/key.txt"
  openssl genpkey -algorithm ED25519 -out "$pem" >/dev/null 2>&1
  openssl pkey -in "$pem" -text -noout >"$key_text"

  python3 - "$key_text" "$PRIVATE_PATH" "$PUBLIC_PATH" <<'PY'
import re
import sys
from pathlib import Path
import os

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

priv_out.parent.mkdir(parents=True, exist_ok=True)
priv_out.write_text(priv + "\n")
pub_out.write_text(pub + "\n")
os.chmod(priv_out, 0o600)
os.chmod(pub_out, 0o644)
PY

  rm -rf "$tmpdir"
}

enforce_permissions() {
  chmod 600 "$PRIVATE_PATH" 2>/dev/null || true
  chmod 644 "$PUBLIC_PATH" 2>/dev/null || true
  chmod 644 "$COMMENT_PATH" 2>/dev/null || true
  chmod 644 "$TIMESTAMP_PATH" 2>/dev/null || true
}

validate_or_generate() {
  if [ -f "$PRIVATE_PATH" ] && [ -f "$PUBLIC_PATH" ]; then
    local priv_hex pub_hex
    priv_hex="$(read_key_hex "$PRIVATE_PATH")"
    pub_hex="$(read_key_hex "$PUBLIC_PATH")"
    if is_hex_len "$priv_hex" 64 && is_hex_len "$pub_hex" 64; then
      return 0
    fi
    echo "error: canonical key files exist but are invalid: $PRIVATE_PATH $PUBLIC_PATH" >&2
    exit 2
  fi

  if command -v openssl >/dev/null 2>&1; then
    generate_with_openssl
    return 0
  fi

  echo "error: openssl not found; canonical key bootstrap requires openssl for ed25519 generation" >&2
  exit 2
}

validate_or_generate

if [ ! -f "$COMMENT_PATH" ]; then
  mkdir -p "$(dirname "$COMMENT_PATH")"
  printf "%s\n" "$KEY_COMMENT" >"$COMMENT_PATH"
fi

if [ ! -f "$TIMESTAMP_PATH" ]; then
  date -u +"%Y-%m-%dT%H:%M:%SZ" >"$TIMESTAMP_PATH"
fi

enforce_permissions

PRIV_HEX="$(read_key_hex "$PRIVATE_PATH")"
PUB_HEX="$(read_key_hex "$PUBLIC_PATH")"
if ! is_hex_len "$PRIV_HEX" 64 || ! is_hex_len "$PUB_HEX" 64; then
  echo "error: canonical key bootstrap produced invalid key lengths" >&2
  exit 2
fi

CANONICAL_PUBKEY_FINGERPRINT="$(compute_pubkey_fingerprint)"
CANONICAL_BOOTSTRAP_TIMESTAMP="$(tr -d '[:space:]' <"$TIMESTAMP_PATH")"

write_exports "$CANONICAL_PUBKEY_FINGERPRINT" "$CANONICAL_BOOTSTRAP_TIMESTAMP"

export OMEGA_CANONICAL_PRIVKEY="$PRIVATE_PATH"
export OMEGA_CANONICAL_PUBKEY="$PUBLIC_PATH"
export OMEGA_CANONICAL_PUBKEY_FINGERPRINT="$CANONICAL_PUBKEY_FINGERPRINT"
export OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP="$CANONICAL_BOOTSTRAP_TIMESTAMP"
export OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP_PATH="$TIMESTAMP_PATH"
export OMEGA_CANONICAL_KEY_COMMENT_PATH="$COMMENT_PATH"

echo "omega_canonical_key_bootstrap: priv=$PRIVATE_PATH pub=$PUBLIC_PATH fingerprint=$CANONICAL_PUBKEY_FINGERPRINT env=$ENV_OUT"
