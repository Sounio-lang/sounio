#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

TMP_DIR="$(mktemp -d)"
OUT_BIN="$TMP_DIR/lean-token-cap-souc"
SMOKE_SRC="$TMP_DIR/lean-token-cap-smoke.sio"
SMOKE_OUT="$TMP_DIR/lean-token-cap-smoke.out"
LOG="$TMP_DIR/lean-token-cap-smoke.log"
trap 'rm -rf "$TMP_DIR"' EXIT

# Rebuild the canonical lean compiler from source so this smoke exercises the
# current self-hosted frontend rather than any prebuilt launcher artifact.
"$SOUC_BIN" self-hosted/compiler/lean_single.sio "$OUT_BIN"
test -x "$OUT_BIN"

# Generate a tiny-valid program with more than 524288 tokens but minimal
# semantic work: a large run of semicolons inside main that the parser skips.
{
  printf 'fn main() -> i64 {\n'
  head -c 600000 /dev/zero | tr '\0' ';'
  printf '\n0\n}\n'
} > "$SMOKE_SRC"

if ! "$OUT_BIN" "$SMOKE_SRC" "$SMOKE_OUT" >"$LOG" 2>&1; then
  tail -n 80 "$LOG" >&2 || true
  exit 1
fi

TOKENS="$(sed -n 's/^tokens: //p' "$LOG" | head -n 1 | tr -d '[:space:]')"
if [[ -z "$TOKENS" ]]; then
  echo "error: token-cap smoke could not read compiler token count" >&2
  tail -n 40 "$LOG" >&2 || true
  exit 1
fi

if (( TOKENS <= 524288 )); then
  echo "error: token-cap smoke stayed at $TOKENS tokens, expected > 524288" >&2
  tail -n 40 "$LOG" >&2 || true
  exit 1
fi

test -s "$SMOKE_OUT"

echo "LEAN_TOKEN_CAP_SMOKE_PASS tokens=$TOKENS"
