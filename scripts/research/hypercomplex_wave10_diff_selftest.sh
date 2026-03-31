#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

ARTIFACT_DIR="${HYPERCOMPLEX_WAVE10_DIFF_DIR:-$ROOT_DIR/artifacts/research}"
mkdir -p "$ARTIFACT_DIR"

FIXTURE_PATH="tests/research/hypercomplex/wave10_hyper_expr_law_profile_audit_compile_proof.sio"
LOG_PATH="$ARTIFACT_DIR/hypercomplex_wave10_diff_selftest.log"
OUT_PATH="$ARTIFACT_DIR/hypercomplex_wave10_diff_selftest.elf"

"$SOUC_BIN" compile "$FIXTURE_PATH" -o "$OUT_PATH" >"$LOG_PATH" 2>&1

if git show hypercomplex-algebra-wave9-codex:self-hosted/check/mod.sio | grep -q "check_hyper_expr_law_profile_audit_tag"; then
    echo "[hypercomplex-wave10] baseline unexpectedly already had check_hyper_expr_law_profile_audit_tag" >&2
    exit 1
fi

for path in \
    self-hosted/check/mod.sio \
    self-hosted/compiler/main.sio
do
    if ! grep -Fq "check_hyper_expr_law_profile_audit_tag" "$ROOT_DIR/$path"; then
        echo "[hypercomplex-wave10] current branch missing check_hyper_expr_law_profile_audit_tag in $path" >&2
        exit 1
    fi
done

if ! grep -Fq "HYPER_WAVE10_AUDIT_CONSUMER_OK" "$ROOT_DIR/$FIXTURE_PATH"; then
    echo "[hypercomplex-wave10] fixture missing Wave 10 success marker" >&2
    exit 1
fi

if ! grep -q "compile: fns=" "$LOG_PATH"; then
    cat "$LOG_PATH"
    echo "[hypercomplex-wave10] baseline comparison compile log missing compile marker" >&2
    exit 1
fi

echo "[hypercomplex-wave10] baseline comparison passed: $LOG_PATH"
