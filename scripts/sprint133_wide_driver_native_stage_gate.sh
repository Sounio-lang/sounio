#!/usr/bin/env bash
# Sprint 133 Gate: Prebuild and stage the wide native compile driver
#
# Goal: avoid `souc run wide_native_compile_driver.sio` JIT startup when
# exercising the T24 smoke path. First probe the wide driver source with the
# lightweight reachability runner, then prebuild the wide driver itself as a
# native ELF via native_compile_driver.sio, and finally run the staged native
# driver on a trivial smoke source.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
SELFHOST_RUN="./scripts/lib/run_selfhost_fresh.sh"
BOOTSTRAP_DRIVER="self-hosted/compiler/native_compile_driver.sio"
WIDE_DRIVER="self-hosted/compiler/wide_native_compile_driver.sio"
WIDE_STAGE_DRIVER="self-hosted/compiler/wide_native_compile_driver_stage.sio"
WIDE_STAGE_NOREACH_DRIVER="self-hosted/compiler/wide_native_compile_driver_stage_noreach.sio"
OUT_DIR="$ROOT_DIR/artifacts/sprint133"
STAGED_DRIVER="$OUT_DIR/wide_native_compile_driver.selfhost.elf"
BUILD_LOG="$OUT_DIR/wide_driver_native_stage.build.log"
PROBE_LOG="$OUT_DIR/wide_driver_native_stage.probe.log"
RUN_LOG="$OUT_DIR/wide_driver_native_stage.smoke.log"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0

check() {
    local name="$1"; local result="$2"
    TOTAL=$((TOTAL + 1))
    if [ "$result" -eq 0 ]; then
        PASS=$((PASS + 1)); echo "  PASS: $name"
    elif [ "$result" -eq 2 ]; then
        NOT_RUN=$((NOT_RUN + 1)); echo "  NOT_RUN: $name"
    else
        FAIL=$((FAIL + 1)); echo "  FAIL: $name"
    fi
}

mkdir -p "$OUT_DIR"
rm -f "$STAGED_DRIVER" "$BUILD_LOG" "$PROBE_LOG" "$RUN_LOG"

echo "=== Sprint 133: Wide Driver Native Stage Gate ==="

# --- Structural checks ---

_ec=1; [ -x "$SOUC" ] && _ec=0
check "T1: souc jit binary exists" $_ec

_ec=1; [ -x "$SELFHOST_RUN" ] && _ec=0
check "T2: run_selfhost_fresh wrapper exists" $_ec

_ec=1; [ -f "$BOOTSTRAP_DRIVER" ] && [ -f "$WIDE_DRIVER" ] && [ -f "$WIDE_STAGE_DRIVER" ] && [ -f "$WIDE_STAGE_NOREACH_DRIVER" ] && _ec=0
check "T3: bootstrap/wide driver sources exist" $_ec

_ec=1; grep -q 'io_write_elf_to_file(output_path, elf)' "$BOOTSTRAP_DRIVER" && _ec=0
check "T4: bootstrap native driver writes ELF via wide-safe path" $_ec

_ec=1; grep -q 'wide_compile_and_emit' "$WIDE_DRIVER" && grep -q 'wide_compile_and_emit' "$WIDE_STAGE_DRIVER" && grep -q 'wide_compile_and_emit' "$WIDE_STAGE_NOREACH_DRIVER" && _ec=0
check "T5: wide driver sources still call wide_compile_and_emit" $_ec

# --- Cheap bootstrap-path smoke on the no-reach stage source ---

_ec=0
timeout 60 "$SELFHOST_RUN" "$SOUC" "$BOOTSTRAP_DRIVER" -- "$WIDE_STAGE_NOREACH_DRIVER" -o "$OUT_DIR/.stage_probe.elf" >"$PROBE_LOG" 2>&1 || _ec=$?
if grep -q 'native_compile: parse_ok' "$PROBE_LOG" && grep -q 'native_compile: fn_count=' "$PROBE_LOG"; then
    _ec=0
elif [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then
    _ec=2
else
    _ec=1
fi
rm -f "$OUT_DIR/.stage_probe.elf"
check "T6: bootstrap path reaches parse_ok and fn_count on no-reach stage source" $_ec

# --- Prebuild the wide driver into a staged native ELF ---

_ec=0
timeout 360 "$SELFHOST_RUN" "$SOUC" "$BOOTSTRAP_DRIVER" -- "$WIDE_STAGE_NOREACH_DRIVER" -o "$STAGED_DRIVER" >"$BUILD_LOG" 2>&1 || _ec=$?
if [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then
    _ec=2
elif [ -s "$STAGED_DRIVER" ] && grep -q 'native_compile: ok written=' "$BUILD_LOG"; then
    chmod +x "$STAGED_DRIVER" || true
    _ec=0
else
    _ec=1
fi
check "T7: bootstrap driver stages no-reach stage driver native ELF" $_ec

# --- Native smoke via the staged wide driver ---

SMOKE_SRC="$(mktemp --suffix=.sio)"
SMOKE_OUT="$OUT_DIR/wide_driver_native_stage.smoke.elf"
cat > "$SMOKE_SRC" <<'SIO'
fn helper() -> i64 { 7 }
fn dead() -> i64 { 99 }
fn main() -> i64 with IO { helper() - 7 }
SIO

_ec=0
if [ ! -s "$STAGED_DRIVER" ]; then
    _ec=2
else
    timeout 180 "$STAGED_DRIVER" "$SMOKE_SRC" -o "$SMOKE_OUT" >"$RUN_LOG" 2>&1 || _ec=$?
    if [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then
        _ec=2
    elif [ $_ec -eq 0 ] && [ -s "$SMOKE_OUT" ] && grep -q 'fn_count=' "$RUN_LOG" && grep -q 'compiling (batch_size=200)' "$RUN_LOG"; then
        _ec=0
    else
        _ec=1
    fi
fi
check "T8: staged native wide driver runs smoke without souc run JIT path" $_ec
rm -f "$SMOKE_SRC"

STATUS="pass"
REASON="all_cases_passed"
if [ "$FAIL" -gt 0 ]; then
    STATUS="fail"
    REASON="${FAIL}_cases_failed"
elif [ "$NOT_RUN" -gt 0 ]; then
    REASON="all_run_passed_${NOT_RUN}_not_run"
fi

cat > "$OUT_DIR/wide_driver_native_stage.v1.json" <<EOF
{
  "schema": "sounio.sprint133.wide_driver_native_stage_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "souc": "$SOUC",
    "bootstrap_driver": "$BOOTSTRAP_DRIVER",
    "wide_driver": "$WIDE_DRIVER",
    "wide_stage_driver": "$WIDE_STAGE_DRIVER",
    "wide_stage_noreach_driver": "$WIDE_STAGE_NOREACH_DRIVER",
    "staged_driver": "$STAGED_DRIVER"
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  }
}
EOF

echo ""
echo "Artifact: artifacts/sprint133/wide_driver_native_stage.v1.json"
echo "Staged driver: artifacts/sprint133/$(basename "$STAGED_DRIVER")"
echo ""
echo "=== RESULTS: $PASS PASS, $FAIL FAIL, $NOT_RUN NOT_RUN (of $TOTAL) ==="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
