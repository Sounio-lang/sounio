#!/usr/bin/env bash
# sprint61_skills_coverage_gate.sh — Full Skills Coverage Gate
#
# Validates:
#   1. All 15 required skills exist with valid structure (via validate_skills_coverage.sh)
#   2. Cross-skill index covers all 15 skills
#   3. model-dispatch routing has >= 3 new rows (added in Sprint 60)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file '$file' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' not found in $file)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 61: Skills Coverage Gate ==="
echo ""

# --- Group 1: validate_skills_coverage.sh inline run ---
echo "--- Group 1: All 15 skills valid ---"
COVERAGE_OUTPUT="$(bash scripts/validate_skills_coverage.sh 2>&1 || true)"
echo "$COVERAGE_OUTPUT"
echo ""

TOTAL=$((TOTAL+1))
if echo "$COVERAGE_OUTPUT" | grep -q "^FAIL"; then
    COVERAGE_FAIL=$(echo "$COVERAGE_OUTPUT" | grep -c "^FAIL")
    echo "FAIL  coverage:all_15_skills_valid ($COVERAGE_FAIL skills failed)"; FAIL=$((FAIL+1))
else
    echo "PASS  coverage:all_15_skills_valid"; PASS=$((PASS+1))
fi
echo ""

# --- Group 2: cross-skill index completeness ---
echo "--- Group 2: Cross-skill index completeness ---"
TOTAL=$((TOTAL+1))
INDEX_FILE="skills/model-dispatch/references/cross-skill-index.md"
if [ ! -f "$INDEX_FILE" ]; then
    echo "FAIL  index:exists (cross-skill-index.md missing)"; FAIL=$((FAIL+1))
else
    INDEX_SKILL_REFS=$(grep -c "sounio-" "$INDEX_FILE" 2>/dev/null || echo 0)
    if [ "$INDEX_SKILL_REFS" -ge 15 ]; then
        echo "PASS  index:has_15_skills ($INDEX_SKILL_REFS sounio-* references found)"; PASS=$((PASS+1))
    else
        echo "FAIL  index:has_15_skills (got $INDEX_SKILL_REFS, want >= 15)"; FAIL=$((FAIL+1))
    fi
fi

# Check all three new skills are in the index
check_grep "index:has_sounio_render"     "sounio-render"     skills/model-dispatch/references/cross-skill-index.md
check_grep "index:has_sounio_bootstrap"  "sounio-bootstrap"  skills/model-dispatch/references/cross-skill-index.md
check_grep "index:has_sounio_pgo"        "sounio-pgo"        skills/model-dispatch/references/cross-skill-index.md
echo ""

# --- Group 3: model-dispatch routing completeness ---
echo "--- Group 3: Model-dispatch routing rows ---"
DISPATCH_FILE="skills/model-dispatch/SKILL.md"
TOTAL=$((TOTAL+1))
NEW_ROWS=$(grep -cE "render example|rasterize|native-compile|PGO pass|sprint gate|SVG" "$DISPATCH_FILE" 2>/dev/null || echo 0)
if [ "$NEW_ROWS" -ge 3 ]; then
    echo "PASS  dispatch:has_3_plus_new_rows ($NEW_ROWS matching rows found)"; PASS=$((PASS+1))
else
    echo "FAIL  dispatch:has_3_plus_new_rows (got $NEW_ROWS, want >= 3)"; FAIL=$((FAIL+1))
fi

check_grep "dispatch:flowchart_has_render"    "Render example completion"    "$DISPATCH_FILE"
check_grep "dispatch:flowchart_has_pgo"       "PGO pass"                     "$DISPATCH_FILE"
check_grep "dispatch:flowchart_has_docs"      "Website/docs/skills"          "$DISPATCH_FILE"
echo ""

# --- Summary ---
echo "=== Sprint 61 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"
echo ""

if [ "$FAIL" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
    echo "STATUS: PASS"
    exit 0
else
    echo "STATUS: FAIL"
    exit 1
fi
