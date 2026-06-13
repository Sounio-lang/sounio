#!/usr/bin/env bash
# Build the Stage1 modular Sounio compiler (Madaros) from
# self-hosted/compiler/main.sio.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT="${1:-$ROOT_DIR/artifacts/self-hosted/madaros}"
SRC="$ROOT_DIR/self-hosted/compiler/main.sio"

resolve_seed() {
  local cand
  for cand in \
    "${SOUC_NATIVE_BIN:-}" \
    "${SOUNIO_SOUC_RAW_BIN:-}" \
    "${SOUC_BIN:-}" \
    "${SOUNIO_SOUC_BIN:-}" \
    "$ROOT_DIR/bin/souc-linux-x86_64" \
    "$ROOT_DIR/bin/souc"; do
    if [[ -n "$cand" && -x "$cand" ]] && [[ "$(head -c 2 "$cand" 2>/dev/null || true)" != "#!" ]]; then
      echo "$cand"
      return 0
    fi
  done
  echo "error: build_modular_madaros: no raw Stage0 seed compiler found" >&2
  return 1
}

[[ -f "$SRC" ]] || {
  echo "error: modular compiler source not found: $SRC" >&2
  exit 1
}

SEED="$(resolve_seed)"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

echo "Building Madaros (Stage1 modular compiler):"
echo "  seed: $SEED"
echo "  src:  $SRC"
echo "  out:  $OUT"

scripts/dev/souc-build-lock.sh "$SEED" "$SRC" "$OUT"

[[ -s "$OUT" ]] || {
  echo "error: modular compiler build produced no output: $OUT" >&2
  exit 1
}

chmod +x "$OUT"
echo "Madaros ready: $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"
