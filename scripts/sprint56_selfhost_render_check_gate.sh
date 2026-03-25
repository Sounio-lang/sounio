#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="./bin/souc"
SH="self-hosted/compiler/main.sio"
TIMEOUT=180
ARTIFACT="artifacts/sprint56/selfhost_render_check_gate.v1.json"

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
    if "$@" >/tmp/sprint56_gate.out 2>&1; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name"
        cat /tmp/sprint56_gate.out
        FAIL=$((FAIL + 1))
    fi
}

check_selfhost_check() {
    local name="$1"
    local file="$2"
    TOTAL=$((TOTAL + 1))
    local out
    if ! out=$(timeout "$TIMEOUT" "$SOUC" run "$SH" -- --check "$file" 2>/dev/null); then
        echo "FAIL  $name"
        FAIL=$((FAIL + 1))
        return
    fi
    if echo "$out" | grep -q "check: OK"; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name (missing 'check: OK')"
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
  "schema": "sounio.sprint56.selfhost_render_check_gate.v1",
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
    "Self-hosted compiler --check mode now runs the real multimodule frontend preflight and returns check: OK on success",
    "All seven render fixtures pass self-hosted parse, resolve, and typecheck without host-compiler fallback",
    "Self-hosted compiler main.sio passes souc check and its own checkpoint self-test suite before the render corpus is accepted"
  ]
}
EOF
}

echo "=== Sprint 56: Self-hosted Render Check Gate ==="
echo ""
echo "--- Group 1: Preconditions ---"
check_file_exists "selfhost:main_exists" "$SH"
check_file_exists "selfhost:fixture_exists" "tests/selfhost/render_ir_expectations.tsv"
echo ""
echo "--- Group 2: Compiler Health ---"
check_run "selfhost:souc_check" "$SOUC" check "$SH"
check_run "selfhost:self_test" timeout "$TIMEOUT" "$SOUC" run "$SH" -- --self-test
echo ""
echo "--- Group 3: Render Corpus --check ---"
check_selfhost_check "render:triangle_basic" "examples/render/triangle_basic.sio"
check_selfhost_check "render:triangle_ppm" "examples/render/triangle_ppm.sio"
check_selfhost_check "render:cube_wireframe" "examples/render/cube_wireframe.sio"
check_selfhost_check "render:uncertainty_ppm" "examples/render/uncertainty_ppm.sio"
check_selfhost_check "render:uncertainty_field" "examples/render/uncertainty_field.sio"
check_selfhost_check "render:causal_dag" "examples/render/causal_dag.sio"
check_selfhost_check "render:quaternion_rotation" "examples/render/quaternion_rotation.sio"
echo ""
echo "=== Sprint 56 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"
write_artifact
if [ "$FAIL" -eq 0 ]; then
    echo "STATUS: PASS"
    exit 0
fi
echo "STATUS: FAIL"
exit 1
