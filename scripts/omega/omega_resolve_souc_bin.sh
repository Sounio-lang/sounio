#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SOUC_VERSION="${SOUNIO_SOUC_VERSION:-0.100.3}"
SOUC_PLATFORM="${SOUNIO_SOUC_PLATFORM:-linux-x86_64}"
SOUC_RELEASE_BASE_URL="${SOUNIO_SOUC_RELEASE_BASE_URL:-https://github.com/sounio-lang/sounio/releases/download}"
SOUC_ASSET_NAME="${SOUNIO_SOUC_ASSET_NAME:-souc-${SOUC_PLATFORM}}"
SOUC_CACHE_DIR="${SOUNIO_SOUC_CACHE_DIR:-$ROOT_DIR/artifacts/omega/souc-bin}"
SOUC_EXPECTED_SHA256="${SOUNIO_SOUC_SHA256:-}"
SOUC_REQUIRE_PINNED="${OMEGA_SOUC_REQUIRE_PINNED:-1}"
ALLOW_LOCAL_FALLBACK="${OMEGA_SOUC_ALLOW_LOCAL_FALLBACK:-0}"
CANONICAL_PUBKEY="${OMEGA_CANONICAL_PUBKEY:-$ROOT_DIR/keys/bootstrap_ed25519.pub}"
PROVENANCE_OUT="${OMEGA_SOUC_PROVENANCE_OUT:-$ROOT_DIR/artifacts/omega/souc_release_provenance.v1.json}"
PRINT_PATH=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --print-path)
      PRINT_PATH=1
      shift
      ;;
    --require-pinned)
      SOUC_REQUIRE_PINNED=1
      shift
      ;;
    --allow-local-fallback)
      ALLOW_LOCAL_FALLBACK=1
      shift
      ;;
    --deny-local-fallback)
      ALLOW_LOCAL_FALLBACK=0
      shift
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

resolve_local() {
  local c
  for c in \
    "${SOUC_BIN:-}" \
    "$ROOT_DIR/souc" \
    "$ROOT_DIR/target/release/souc" \
    "$ROOT_DIR/target/debug/souc"; do
    if [ -n "$c" ] && [ -x "$c" ]; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  if command -v souc >/dev/null 2>&1; then
    command -v souc
    return 0
  fi
  return 1
}

mkdir -p "$SOUC_CACHE_DIR"

if [ "$SOUC_VERSION" = "latest" ]; then
  ASSET_URL="${SOUNIO_SOUC_ASSET_URL:-}"
else
  ASSET_URL="${SOUNIO_SOUC_ASSET_URL:-$SOUC_RELEASE_BASE_URL/v${SOUC_VERSION}/${SOUC_ASSET_NAME}}"
fi

BIN_PATH="$SOUC_CACHE_DIR/$SOUC_ASSET_NAME"
SHA_PATH="$SOUC_CACHE_DIR/$SOUC_ASSET_NAME.sha256"
SIG_PATH="$SOUC_CACHE_DIR/$SOUC_ASSET_NAME.sig"

resolved=""
if [ "$SOUC_REQUIRE_PINNED" = "1" ] && [ -n "$ASSET_URL" ]; then
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$ASSET_URL" -o "$BIN_PATH"
    chmod +x "$BIN_PATH"

    if [ -z "$SOUC_EXPECTED_SHA256" ]; then
      set +e
      curl -fsSL "${ASSET_URL}.sha256" -o "$SHA_PATH"
      sha_rc=$?
      set -e
      if [ "$sha_rc" -eq 0 ] && [ -s "$SHA_PATH" ]; then
        SOUC_EXPECTED_SHA256="$(awk '{print $1}' "$SHA_PATH" | tr -d '[:space:]')"
      fi
    fi

    set +e
    curl -fsSL "${ASSET_URL}.sig" -o "$SIG_PATH"
    sig_rc=$?
    set -e

    VERIFY_ARGS=(
      --binary "$BIN_PATH"
      --out "$PROVENANCE_OUT"
      --version "$SOUC_VERSION"
      --platform "$SOUC_PLATFORM"
      --asset-url "$ASSET_URL"
      --public-key "$CANONICAL_PUBKEY"
      --require-signature
    )

    if [ -n "$SOUC_EXPECTED_SHA256" ]; then
      VERIFY_ARGS+=(--sha256 "$SOUC_EXPECTED_SHA256")
    elif [ -s "$SHA_PATH" ]; then
      VERIFY_ARGS+=(--sha256-file "$SHA_PATH")
    fi

    if [ "$sig_rc" -eq 0 ] && [ -s "$SIG_PATH" ]; then
      VERIFY_ARGS+=(--signature-file "$SIG_PATH")
    fi

    python3 "$ROOT_DIR/scripts/omega/omega_verify_release_souc.py" "${VERIFY_ARGS[@]}" 1>&2
    resolved="$BIN_PATH"
  else
    echo "error: curl required for pinned souc resolution" >&2
    exit 2
  fi
fi

if [ "$SOUC_REQUIRE_PINNED" = "1" ] && [ -z "$ASSET_URL" ]; then
  if [ "$ALLOW_LOCAL_FALLBACK" != "1" ]; then
    echo "error: pinned souc resolution requires SOUNIO_SOUC_ASSET_URL or SOUNIO_SOUC_VERSION (not 'latest')" >&2
    exit 2
  fi
  echo "warning: pinned souc requested but asset URL unresolved; using local fallback" >&2
fi

if [ -z "$resolved" ]; then
  if [ "$ALLOW_LOCAL_FALLBACK" = "1" ]; then
    if ! resolved="$(resolve_local)"; then
      echo "error: unable to resolve local souc binary" >&2
      exit 2
    fi
    # Best-effort provenance in fallback mode.
    python3 "$ROOT_DIR/scripts/omega/omega_verify_release_souc.py" \
      --binary "$resolved" \
      --out "$PROVENANCE_OUT" \
      --version "local-fallback" \
      --platform "$SOUC_PLATFORM" \
      --asset-url "" \
      --public-key "$CANONICAL_PUBKEY" >/dev/null || true
  else
    echo "error: pinned souc required but resolution failed" >&2
    exit 2
  fi
fi

if [ "$PRINT_PATH" = "1" ]; then
  printf '%s\n' "$resolved"
else
  echo "SOUC_BIN=$resolved"
fi
