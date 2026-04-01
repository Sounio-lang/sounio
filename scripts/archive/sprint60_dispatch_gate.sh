#!/usr/bin/env bash
# sprint60_dispatch_gate.sh — Model Dispatch Update + Cross-Skill Index
#
# Validates that model-dispatch/SKILL.md has the 7 new routing rows
# and that the cross-skill index covers all 15 skills.
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

check_file_exists() {
    local name="$1"; local file="$2"
    TOTAL=$((TOTAL+1))
    if [ -f "$file" ]; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (file '$file' missing)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 60: Model Dispatch Update + Cross-Skill Index ==="
echo ""

# --- Group 1: model-dispatch/SKILL.md new routing rows ---
echo "--- Group 1: model-dispatch/SKILL.md routing rows ---"
check_grep "dispatch:has_render_row"         "render example"           skills/model-dispatch/SKILL.md
check_grep "dispatch:has_rasterize_row"      "rasterize"                skills/model-dispatch/SKILL.md
check_grep "dispatch:has_check_failure_row"  "check failure|--check"    skills/model-dispatch/SKILL.md
check_grep "dispatch:has_native_compile_row" "native-compile"           skills/model-dispatch/SKILL.md
check_grep "dispatch:has_pgo_row"            "PGO pass"                 skills/model-dispatch/SKILL.md
check_grep "dispatch:has_gate_triage_row"    "sprint gate|gate.*triage" skills/model-dispatch/SKILL.md
check_grep "dispatch:has_svg_preview_row"    "SVG preview|SVG"          skills/model-dispatch/SKILL.md
echo ""

# --- Group 2: model-dispatch/SKILL.md flowchart entries ---
echo "--- Group 2: model-dispatch/SKILL.md flowchart entries ---"
check_grep "dispatch:flowchart_render"       "Render example completion" skills/model-dispatch/SKILL.md
check_grep "dispatch:flowchart_bootstrap"    "self-hosted frontend"      skills/model-dispatch/SKILL.md
check_grep "dispatch:flowchart_pgo"          "PGO pass.*IR module"       skills/model-dispatch/SKILL.md
check_grep "dispatch:flowchart_docs"         "Website/docs/skills"       skills/model-dispatch/SKILL.md
echo ""

# --- Group 3: cross-skill index exists and is complete ---
echo "--- Group 3: cross-skill-index.md ---"
check_file_exists "dispatch:cross_index_exists"   skills/model-dispatch/references/cross-skill-index.md
check_grep "dispatch:cross_index_has_render"      "sounio-render"     skills/model-dispatch/references/cross-skill-index.md
check_grep "dispatch:cross_index_has_bootstrap"   "sounio-bootstrap"  skills/model-dispatch/references/cross-skill-index.md
check_grep "dispatch:cross_index_has_pgo"         "sounio-pgo"        skills/model-dispatch/references/cross-skill-index.md
check_grep "dispatch:cross_index_has_language"    "sounio-language"   skills/model-dispatch/references/cross-skill-index.md
check_grep "dispatch:cross_index_has_codegen"     "sounio-codegen"    skills/model-dispatch/references/cross-skill-index.md
check_grep "dispatch:cross_index_has_primary"     "PRIMARY"           skills/model-dispatch/references/cross-skill-index.md
echo ""

# --- Group 4: cross-skill index has all 15 skills ---
echo "--- Group 4: cross-skill-index.md skill count ---"
TOTAL=$((TOTAL+1))
SKILL_COUNT=$(grep -c "sounio-" skills/model-dispatch/references/cross-skill-index.md 2>/dev/null || echo 0)
if [ "$SKILL_COUNT" -ge 15 ]; then
    echo "PASS  dispatch:cross_index_15_skills ($SKILL_COUNT sounio-* references found)"; PASS=$((PASS+1))
else
    echo "FAIL  dispatch:cross_index_15_skills (got $SKILL_COUNT, want >= 15)"; FAIL=$((FAIL+1))
fi
echo ""

# --- Summary ---
echo "=== Sprint 60 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"
echo ""

if [ "$FAIL" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
    echo "STATUS: PASS"
    exit 0
else
    echo "STATUS: FAIL"
    exit 1
fi
