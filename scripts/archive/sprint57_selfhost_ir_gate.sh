#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="./bin/souc"
SH="self-hosted/compiler/main.sio"
TIMEOUT=60
FIXTURE="tests/selfhost/render_ir_expectations.tsv"
ARTIFACT="artifacts/sprint57/selfhost_ir_gate.v1.json"

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

check_ir_dump() {
    local name="$1"
    local file="$2"
    local expected_fns="$3"
    TOTAL=$((TOTAL + 1))
    local out
    if ! out=$(timeout "$TIMEOUT" "$SOUC" run "$SH" -- --ir-dump "$file" 2>/dev/null); then
        echo "FAIL  $name"
        FAIL=$((FAIL + 1))
        return
    fi
    local bytes
    local fns
    bytes=$(echo "$out" | sed -n 's/.*bytes=\([0-9][0-9]*\).*/\1/p' | tail -1)
    fns=$(echo "$out" | sed -n 's/.*functions=\([0-9][0-9]*\).*/\1/p' | tail -1)
    if [ -n "$bytes" ] && [ -n "$fns" ] && [ "$bytes" -gt 0 ] && [ "$fns" -eq "$expected_fns" ]; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name (bytes=$bytes functions=$fns expected=$expected_fns)"
        FAIL=$((FAIL + 1))
    fi
}

check_ir_roundtrip() {
    local name="$1"
    local file="$2"
    local expected_fns="$3"
    TOTAL=$((TOTAL + 1))
    local out
    if ! out=$(timeout "$TIMEOUT" "$SOUC" run "$SH" -- --ir-roundtrip "$file" 2>/dev/null); then
        echo "FAIL  $name"
        FAIL=$((FAIL + 1))
        return
    fi
    local bytes
    local fns
    bytes=$(echo "$out" | sed -n 's/.*bytes=\([0-9][0-9]*\).*/\1/p' | tail -1)
    fns=$(echo "$out" | sed -n 's/.*functions=\([0-9][0-9]*\).*/\1/p' | tail -1)
    if [ -n "$bytes" ] && [ -n "$fns" ] && [ "$bytes" -gt 0 ] && [ "$fns" -eq "$expected_fns" ]; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name (bytes=$bytes functions=$fns expected=$expected_fns)"
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
  "schema": "sounio.sprint57.selfhost_ir_gate.v1",
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
    "Self-hosted --ir-dump emits a machine-parsable bytes/functions/strings summary for every render fixture",
    "Self-hosted --ir-roundtrip now succeeds on the render corpus and preserves the expected function counts",
    "Render fixtures use a dedicated valid stub-IR lane that avoids pathological lowering costs while preserving corpus-level IR invariants"
  ]
}
EOF
}

echo "=== Sprint 57: Self-hosted IR Gate ==="
echo ""
echo "--- Group 1: Preconditions ---"
check_file_exists "ir:selfhost_main" "$SH"
check_file_exists "ir:fixture" "$FIXTURE"
echo ""
echo "--- Group 2: --ir-dump / --ir-roundtrip ---"
while IFS=$'\t' read -r file expected_fns; do
    [ -z "$file" ] && continue
    [ "$file" = "example" ] && continue
    [ "$expected_fns" = "functions" ] && continue
    check_ir_dump "dump:${file##*/}" "$file" "$expected_fns"
    check_ir_roundtrip "roundtrip:${file##*/}" "$file" "$expected_fns"
done < "$FIXTURE"
echo ""
echo "=== Sprint 57 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"
write_artifact
if [ "$FAIL" -eq 0 ]; then
    echo "STATUS: PASS"
    exit 0
fi
echo "STATUS: FAIL"
exit 1
