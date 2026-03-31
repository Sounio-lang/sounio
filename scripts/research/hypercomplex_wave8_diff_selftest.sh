#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

ARTIFACT_DIR="${HYPERCOMPLEX_WAVE8_DIFF_DIR:-$ROOT_DIR/artifacts/research}"
mkdir -p "$ARTIFACT_DIR"

LOG_PATH="$ARTIFACT_DIR/hypercomplex_wave8_diff_selftest.log"
OUT_PATH="$ARTIFACT_DIR/hypercomplex_wave8_diff_selftest.elf"

"$SOUC_BIN" compile tests/research/hypercomplex/wave8_hyper_expr_law_profile_diff_selftest.sio -o "$OUT_PATH" >"$LOG_PATH" 2>&1

if git show hypercomplex-algebra-wave7-codex:self-hosted/check/check.sio | grep -q "law_profile_source"; then
    echo "[hypercomplex-wave8] baseline unexpectedly already had law_profile_source in check/check.sio" >&2
    exit 1
fi

for path in \
    self-hosted/check/check.sio \
    self-hosted/check/mod.sio \
    self-hosted/ir/ir.sio \
    self-hosted/ir/serialize.sio
do
    if ! grep -q "law_profile_source" "$ROOT_DIR/$path"; then
        echo "[hypercomplex-wave8] current branch missing law_profile_source in $path" >&2
        exit 1
    fi
done

if ! grep -q "compile: fns=" "$LOG_PATH"; then
    cat "$LOG_PATH"
    echo "[hypercomplex-wave8] baseline comparison compile log missing compile marker" >&2
    exit 1
fi

echo "[hypercomplex-wave8] baseline comparison passed: $LOG_PATH"
