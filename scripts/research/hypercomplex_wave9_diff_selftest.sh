#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

ARTIFACT_DIR="${HYPERCOMPLEX_WAVE9_DIFF_DIR:-$ROOT_DIR/artifacts/research}"
mkdir -p "$ARTIFACT_DIR"

FIXTURE_PATH="tests/research/hypercomplex/wave9_hyper_expr_law_profile_fingerprint_diff_selftest.sio"
LOG_PATH="$ARTIFACT_DIR/hypercomplex_wave9_diff_selftest.log"
OUT_PATH="$ARTIFACT_DIR/hypercomplex_wave9_diff_selftest.elf"

"$SOUC_BIN" compile "$FIXTURE_PATH" -o "$OUT_PATH" >"$LOG_PATH" 2>&1

expected_registry_fingerprint=$(( (3 & 255) + ((1 & 255) << 8) + ((2 & 255) << 16) + ((19 & 255) << 24) + ((1 & 255) << 32) ))
expected_fallback_fingerprint=$(( (4 & 255) + ((1 & 255) << 8) + ((1 & 255) << 16) + ((31 & 255) << 24) ))

if git show hypercomplex-algebra-wave8-codex:self-hosted/check/check.sio | grep -q "law_profile_fingerprint"; then
    echo "[hypercomplex-wave9] baseline unexpectedly already had law_profile_fingerprint in check/check.sio" >&2
    exit 1
fi

for path in \
    self-hosted/check/cayley_dickson.sio \
    self-hosted/check/defs.sio \
    self-hosted/check/check.sio \
    self-hosted/check/mod.sio \
    self-hosted/ir/algebra.sio \
    self-hosted/ir/ir.sio \
    self-hosted/ir/serialize.sio \
    self-hosted/compiler/main.sio
do
    if ! grep -Fq "law_profile_fingerprint" "$ROOT_DIR/$path"; then
        echo "[hypercomplex-wave9] current branch missing law_profile_fingerprint in $path" >&2
        exit 1
    fi
done

for path in \
    self-hosted/check/cayley_dickson.sio \
    self-hosted/ir/algebra.sio
do
    for pattern in \
        "let packed_algebra = algebra_tag & 255" \
        "let packed_op = (op_kind & 255) << 8" \
        "let packed_reassoc = (reassoc_strategy & 255) << 16" \
        "let packed_forbidden = (forbidden_law_mask & 255) << 24" \
        "let packed_source = (law_profile_source & 255) << 32" \
        "packed_algebra + packed_op + packed_reassoc + packed_forbidden + packed_source"
    do
        if ! grep -Fq "$pattern" "$ROOT_DIR/$path"; then
            echo "[hypercomplex-wave9] fingerprint formula drift in $path: missing '$pattern'" >&2
            exit 1
        fi
    done
done

if ! grep -Fq "let expected_registry_fingerprint: i64 = $expected_registry_fingerprint" "$ROOT_DIR/$FIXTURE_PATH"; then
    echo "[hypercomplex-wave9] fixture missing expected registry fingerprint constant" >&2
    exit 1
fi

if ! grep -Fq "let expected_fallback_fingerprint: i64 = $expected_fallback_fingerprint" "$ROOT_DIR/$FIXTURE_PATH"; then
    echo "[hypercomplex-wave9] fixture missing expected fallback fingerprint constant" >&2
    exit 1
fi

if ! grep -Fq "HYPER_WAVE9_DIFF_SELFTEST_OK" "$ROOT_DIR/$FIXTURE_PATH"; then
    echo "[hypercomplex-wave9] fixture missing Wave 9 success marker" >&2
    exit 1
fi

if ! grep -q "compile: fns=" "$LOG_PATH"; then
    cat "$LOG_PATH"
    echo "[hypercomplex-wave9] baseline comparison compile log missing compile marker" >&2
    exit 1
fi

echo "[hypercomplex-wave9] baseline comparison passed: $LOG_PATH"
