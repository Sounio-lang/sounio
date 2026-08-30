#!/usr/bin/env bash
# Gate: receiver layout must remain part of field identity through large
# array-of-struct returns, tuple projection, nested indexing, and module edges.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fail() {
    echo "[aggregate-field-identity] FAIL: $*" >&2
    exit 1
}

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
    echo "[aggregate-field-identity] SKIP: Linux-only native witness"
    exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-aggregate-field-identity.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

DRIVER_SRC="$ROOT/self-hosted/compiler/aggregate_field_identity_driver.sio"
WITNESS_SRC="$ROOT/tests/diagnose/aggregate_field_identity_main.sio"
WITNESS_BUILDER="$ROOT/tests/diagnose/aggregate_field_identity_builder.sio"
WITNESS_BOUNDARY="$ROOT/tests/diagnose/aggregate_field_identity_boundary.sio"
LOWER_SRC="$ROOT/self-hosted/ir/lower.sio"
DRIVER_BIN="${SOUNIO_AGGREGATE_FIELD_DRIVER_BIN:-$WORK/aggregate-field-driver}"
FIXED_WITNESS_BIN="/tmp/sounio-aggregate-field-identity-current-source"
WITNESS_BIN="$WORK/aggregate-field-witness"

[[ -f "$DRIVER_SRC" ]] || fail "missing driver source: $DRIVER_SRC"
[[ -f "$WITNESS_SRC" ]] || fail "missing witness source: $WITNESS_SRC"
[[ -f "$WITNESS_BUILDER" ]] || fail "missing witness builder: $WITNESS_BUILDER"
[[ -f "$WITNESS_BOUNDARY" ]] || fail "missing witness boundary: $WITNESS_BOUNDARY"
[[ -f "$LOWER_SRC" ]] || fail "missing lowerer source: $LOWER_SRC"

ulimit -S -s 524288 2>/dev/null || ulimit -S -s unlimited 2>/dev/null || true

resolve_bootstrap_elf() {
    local candidate
    for candidate in "${SOUNIO_AGGREGATE_FIELD_BOOTSTRAP:-}" \
        "$ROOT/bin/souc-linux-x86_64" \
        "$ROOT/bin/souc-lean-single-x86_64"; do
        if [[ -n "$candidate" && -x "$candidate" && "$(head -c2 "$candidate" 2>/dev/null || true)" != '#!' ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

if [[ -n "${SOUNIO_AGGREGATE_FIELD_DRIVER_BIN:-}" ]]; then
    [[ -x "$DRIVER_BIN" ]] || fail "driver override is not executable: $DRIVER_BIN"
    driver_source="override"
else
    SEED="${SOUNIO_AGGREGATE_FIELD_SEED:-}"
    if [[ -n "$SEED" ]]; then
        [[ -x "$SEED" ]] || fail "seed override is not executable: $SEED"
        seed_source="override"
    else
        BOOTSTRAP="$(resolve_bootstrap_elf)" || fail "no bootstrap ELF found"
        SEED="$WORK/source-tracking-lean-seed"
        seed_source="derived_current_lean_single"
        if ! "$ROOT/scripts/dev/souc-build-lock.sh" \
            "$BOOTSTRAP" \
            "$ROOT/self-hosted/compiler/lean_single.sio" \
            "$SEED" >"$WORK/seed.log" 2>&1; then
            tail -n 100 "$WORK/seed.log" >&2 || true
            fail "source-tracking lean seed build failed"
        fi
        chmod +x "$SEED"
    fi

    if ! "$ROOT/scripts/dev/souc-build-lock.sh" \
        "$SEED" "$DRIVER_SRC" "$DRIVER_BIN" >"$WORK/driver-build.log" 2>&1; then
        tail -n 100 "$WORK/driver-build.log" >&2 || true
        fail "current-source aggregate driver build failed"
    fi
    [[ -s "$DRIVER_BIN" ]] || fail "driver build produced no output"
    chmod +x "$DRIVER_BIN"
    driver_source="current_source:$seed_source"
fi

echo "[aggregate-field-identity] driver_source=$driver_source"
echo "[aggregate-field-identity] driver_sha256=$(sha256sum "$DRIVER_BIN" | awk '{print $1}')"
echo "[aggregate-field-identity] driver_source_sha256=$(sha256sum "$DRIVER_SRC" | awk '{print $1}')"
echo "[aggregate-field-identity] lower_source_sha256=$(sha256sum "$LOWER_SRC" | awk '{print $1}')"
echo "[aggregate-field-identity] witness_main_sha256=$(sha256sum "$WITNESS_SRC" | awk '{print $1}')"
echo "[aggregate-field-identity] witness_builder_sha256=$(sha256sum "$WITNESS_BUILDER" | awk '{print $1}')"
echo "[aggregate-field-identity] witness_boundary_sha256=$(sha256sum "$WITNESS_BOUNDARY" | awk '{print $1}')"

command -v flock >/dev/null 2>&1 || fail "flock is required for the fixed-output driver"
BUILD_LOCK="${SOUNIO_BUILD_LOCK:-/tmp/sounio-souc-build.lock}"
exec 8>"$BUILD_LOCK"
flock 8 || fail "could not acquire compiler build lock: $BUILD_LOCK"
rm -f "$FIXED_WITNESS_BIN"
set +e
"$DRIVER_BIN" >"$WORK/witness-build.log" 2>&1
driver_rc=$?
set -e
if [[ "$driver_rc" == "0" && -s "$FIXED_WITNESS_BIN" ]]; then
    cp "$FIXED_WITNESS_BIN" "$WITNESS_BIN"
fi
rm -f "$FIXED_WITNESS_BIN"
flock -u 8
exec 8>&-

if [[ "$driver_rc" != "0" ]]; then
    cat "$WORK/witness-build.log" >&2
    fail "current-source driver could not compile the witness (rc=$driver_rc)"
fi
[[ -s "$WITNESS_BIN" ]] || fail "driver produced no witness ELF"
chmod +x "$WITNESS_BIN"

set +e
"$WITNESS_BIN" >"$WORK/witness.stdout" 2>"$WORK/witness.stderr"
witness_rc=$?
set -e

cat "$WORK/witness.stdout"
if [[ -s "$WORK/witness.stderr" ]]; then
    cat "$WORK/witness.stderr" >&2
fi
[[ "$witness_rc" == "0" ]] || fail "witness exited rc=$witness_rc"
[[ "$(cat "$WORK/witness.stdout")" == "AGGREGATE_FIELD_IDENTITY_OK" ]] || \
    fail "unexpected witness output"

echo "[aggregate-field-identity] PASS: FieldIdentity=(ReceiverLayout,Field) across direct, nested, and transitive reads"
