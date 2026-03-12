#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
SELFHOST_RUN="./scripts/lib/run_selfhost_fresh.sh"
SH="self-hosted/compiler/main.sio"
SUMMARY_SMOKE="self-hosted/compiler/render_streaming_body_smoke.sio"
TIMEOUT=480
OUT_BIN="/tmp/selfhost_triangle_basic_bootstrap"
REF_PPM="/tmp/selfhost_triangle_basic_ref.ppm"
OUT_PPM="/tmp/selfhost_triangle_basic_out.ppm"
ARTIFACT="artifacts/sprint58/selfhost_native_render_gate.v1.json"
TRACE_ARTIFACT="artifacts/omega/native_decision_trace.v1.json"

PASS=0
FAIL=0
NOT_RUN=0
TOTAL=0

check_file_exists() {
    local name="$1"
    local file="$2"
    TOTAL=$((TOTAL + 1))
    if [ -f "$file" ]; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name (missing $file)"
        FAIL=$((FAIL + 1))
    fi
}

check_run() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    if "$@" >/tmp/sprint58_gate.out 2>&1; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name"
        cat /tmp/sprint58_gate.out
        FAIL=$((FAIL + 1))
    fi
}

check_absent_grep() {
    local name="$1"
    local pattern="$2"
    local file="$3"
    TOTAL=$((TOTAL + 1))
    if grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "FAIL  $name (unexpected pattern '$pattern' in $file)"
        FAIL=$((FAIL + 1))
    else
        echo "PASS  $name"
        PASS=$((PASS + 1))
    fi
}

check_native_compile() {
    TOTAL=$((TOTAL + 1))
    rm -f "$OUT_BIN"
    if timeout "$TIMEOUT" "$SELFHOST_RUN" "$SOUC" "$SH" -- --native-compile examples/render/triangle_basic.sio -o "$OUT_BIN" >/tmp/sprint58_compile.out 2>&1; then
        echo "PASS  native:compile"
        PASS=$((PASS + 1))
    else
        echo "FAIL  native:compile"
        cat /tmp/sprint58_compile.out
        FAIL=$((FAIL + 1))
    fi
}

check_ppm_headers() {
    local name="$1"
    local file="$2"
    TOTAL=$((TOTAL + 1))
    local magic dims maxv
    magic=$(sed -n '1p' "$file")
    dims=$(sed -n '2p' "$file")
    maxv=$(sed -n '3p' "$file")
    if [ "$magic" = "P3" ] && [ "$dims" = "128 128" ] && [ "$maxv" = "255" ]; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name (magic=$magic dims=$dims max=$maxv)"
        FAIL=$((FAIL + 1))
    fi
}

write_artifact() {
    local status="fail"
    local reason="one_or_more_checks_failed"
    if [ "$FAIL" -eq 0 ]; then
        status="pass"
        reason="all_cases_passed"
    fi
    mkdir -p "$(dirname "$ARTIFACT")"
    cat > "$ARTIFACT" <<EOF
{
  "schema": "sounio.sprint58.selfhost_native_render_gate.v1",
  "generated_at": "$(date --iso-8601=seconds)",
  "status": "$status",
  "reason": "$reason",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": $TIMEOUT
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "Self-hosted --native-compile now emits triangle_basic through a fresh native backend build without cached ELF reuse",
    "The bootstrap ELF emits a byte-for-byte identical PPM stream to the JIT render reference",
    "The Sprint 58 proof executes the live native backend end-to-end without staged triangle_basic PPM part playback"
  ]
}
EOF
}

echo "=== Sprint 58: Self-hosted Native Render Gate ==="
echo ""
echo "--- Group 1: Preconditions ---"
check_file_exists "native:selfhost_main" "$SH"
check_file_exists "native:selfhost_runner" "$SELFHOST_RUN"
check_file_exists "native:summary_body_smoke" "$SUMMARY_SMOKE"
check_file_exists "native:render_source" "examples/render/triangle_basic.sio"
check_absent_grep "native:no_cached_elf_path" "triangle_basic_bootstrap_cached\\.elf" "$SH"
check_absent_grep "native:no_cached_compile_helper" "compiler_try_cached_render_native_compile" "$SH"
check_absent_grep "native:no_stage_runtime_helper" "stage_native_runtime_bundle" "scripts/e2e_gate.sh"
echo ""
echo "--- Group 2: Compile And Execute ---"
check_run "native:reference_ppm" bash -lc "$SOUC run examples/render/triangle_basic.sio > \"$REF_PPM\""
check_run "native:summary_body_streaming_smoke" "$SELFHOST_RUN" "$SOUC" "$SUMMARY_SMOKE"
check_native_compile
check_run "native:decision_trace_pipeline" grep -qE '"pipeline":"(thinlink-cas|streaming-direct)"' "$TRACE_ARTIFACT"
check_run "native:binary_exists" test -f "$OUT_BIN"
check_run "native:chmod" chmod +x "$OUT_BIN"
check_run "native:execute_binary" bash -lc "\"$OUT_BIN\" > \"$OUT_PPM\""
check_ppm_headers "native:reference_header" "$REF_PPM"
check_ppm_headers "native:bootstrap_header" "$OUT_PPM"
check_run "native:ppm_byte_match" cmp -s "$REF_PPM" "$OUT_PPM"
echo ""
echo "=== Sprint 58 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"
write_artifact
if [ "$FAIL" -eq 0 ]; then
    echo "STATUS: PASS"
    exit 0
fi
echo "STATUS: FAIL"
exit 1
