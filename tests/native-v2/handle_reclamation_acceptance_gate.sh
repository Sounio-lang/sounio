#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPILER="${SOUNIO_HANDLE_RECLAIM_COMPILER:-}"
OUT_DIR="${SOUNIO_HANDLE_RECLAIM_OUT_DIR:-${TMPDIR:-/tmp}/sounio-handle-reclamation-gate}"
STATUS_JSON="$OUT_DIR/status.v1.json"
CONTROL_SRC="$ROOT/tests/native-v2/handle_reclamation_below_capacity.sio"
EXHAUST_SRC="$ROOT/tests/native-v2/handle_reclamation_exhaustion.sio"

mkdir -p "$OUT_DIR"

write_status() {
    local status="$1" passed="$2" failed="$3" not_run="$4" reason="$5"
    printf '{"status":"%s","metrics":{"total":2,"passed":%s,"failed":%s,"not_run":%s},"reason":"%s"}\n' \
        "$status" "$passed" "$failed" "$not_run" "$reason" > "$STATUS_JSON"
}

if [[ -z "$COMPILER" || ! -x "$COMPILER" ]]; then
    write_status "blocked" 0 0 2 "source_built_compiler_required"
    echo "BLOCKED: set SOUNIO_HANDLE_RECLAIM_COMPILER to a source-built Madaros ELF" >&2
    echo "status_json=$STATUS_JSON" >&2
    exit 2
fi

ulimit -s 524288
COMPILER_SHA="$(sha256sum "$COMPILER" | awk '{print $1}')"
echo "compiler=$COMPILER"
echo "compiler_sha256=$COMPILER_SHA"

compile_fixture() {
    local src="$1" out="$2" log="$3"
    rm -f "$out"
    "$ROOT/scripts/dev/souc-build-lock.sh" "$COMPILER" "$src" -o "$out" >"$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 || ! -s "$out" ]]; then
        return 1
    fi
    chmod +x "$out"
}

CONTROL_ELF="$OUT_DIR/control.elf"
EXHAUST_ELF="$OUT_DIR/exhaustion.elf"
if ! compile_fixture "$CONTROL_SRC" "$CONTROL_ELF" "$OUT_DIR/control.compile.log"; then
    write_status "blocked" 0 0 2 "control_compile_failed"
    echo "BLOCKED: control fixture did not compile" >&2
    exit 2
fi
if ! compile_fixture "$EXHAUST_SRC" "$EXHAUST_ELF" "$OUT_DIR/exhaustion.compile.log"; then
    write_status "blocked" 0 0 2 "exhaustion_compile_failed"
    echo "BLOCKED: exhaustion fixture did not compile" >&2
    exit 2
fi

set +e
timeout 180 "$CONTROL_ELF" >"$OUT_DIR/control.run.log" 2>&1
CONTROL_RC=$?
timeout 180 "$EXHAUST_ELF" >"$OUT_DIR/exhaustion.run.log" 2>&1
EXHAUST_RC=$?
set -e

if [[ $CONTROL_RC -ne 0 ]] || ! grep -q '^HANDLE_CONTROL_OK count=1000000$' "$OUT_DIR/control.run.log"; then
    write_status "fail" 0 2 0 "negative_control_failed"
    echo "FAIL: below-capacity control rc=$CONTROL_RC" >&2
    exit 1
fi

if [[ $EXHAUST_RC -eq 0 ]] && grep -q '^HANDLE_RECLAMATION_OK count=4194304$' "$OUT_DIR/exhaustion.run.log"; then
    write_status "pass" 2 0 0 "reclamation_acceptance_met"
    echo "PASS: managed allocations reclaimed before the lifetime wall"
    echo "status_json=$STATUS_JSON"
    exit 0
fi

write_status "fail" 1 1 0 "reclamation_acceptance_not_met"
if [[ $EXHAUST_RC -eq 182 ]]; then
    echo "POSITIVE_CONTROL_FIRED: exhaustion fixture failed closed with rc=182" >&2
else
    echo "FAIL: exhaustion fixture rc=$EXHAUST_RC, expected current-baseline rc=182 or repaired rc=0" >&2
fi
echo "status_json=$STATUS_JSON" >&2
exit 1
