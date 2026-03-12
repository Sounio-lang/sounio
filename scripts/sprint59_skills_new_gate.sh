#!/usr/bin/env bash
# sprint59_skills_new_gate.sh — Skills Platform Expansion: three new domain skills
#
# Validates that sounio-render, sounio-bootstrap, and sounio-pgo skills
# exist with correct YAML frontmatter, Workflow section, and gate references.
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

echo "=== Sprint 59: New Domain Skills — sounio-render, sounio-bootstrap, sounio-pgo ==="
echo ""

# --- Group 1: sounio-render skill ---
echo "--- Group 1: sounio-render/SKILL.md ---"
check_grep "skill:render_has_name"        "^name:"                  skills/sounio-render/SKILL.md
check_grep "skill:render_has_description" "^description:"           skills/sounio-render/SKILL.md
check_grep "skill:render_has_workflow"    "## Workflow"             skills/sounio-render/SKILL.md
check_grep "skill:render_has_gate_ref"    "sprint53"                skills/sounio-render/SKILL.md
check_grep "skill:render_has_65536"       "65536"                   skills/sounio-render/SKILL.md
check_grep "skill:render_has_ppm_table"   "128.*128"                skills/sounio-render/SKILL.md
check_grep "skill:render_has_gpu_section" "GPU"                     skills/sounio-render/SKILL.md
echo ""

# --- Group 2: sounio-render reference file ---
echo "--- Group 2: sounio-render/references/render-navigation.md ---"
check_file_exists "skill:render_nav_exists"        skills/sounio-render/references/render-navigation.md
check_grep "skill:render_nav_has_modules"   "framebuffer"           skills/sounio-render/references/render-navigation.md
check_grep "skill:render_nav_has_examples"  "triangle_basic"        skills/sounio-render/references/render-navigation.md
check_grep "skill:render_nav_has_ppm_dims"  "192.*192"              skills/sounio-render/references/render-navigation.md
echo ""

# --- Group 3: sounio-bootstrap skill ---
echo "--- Group 3: sounio-bootstrap/SKILL.md ---"
check_grep "skill:bootstrap_has_name"             "^name:"          skills/sounio-bootstrap/SKILL.md
check_grep "skill:bootstrap_has_description"      "^description:"   skills/sounio-bootstrap/SKILL.md
check_grep "skill:bootstrap_has_workflow"         "## Workflow"     skills/sounio-bootstrap/SKILL.md
check_grep "skill:bootstrap_has_native_compile"   "native-compile"  skills/sounio-bootstrap/SKILL.md
check_grep "skill:bootstrap_has_check_mode"       "\-\-check"       skills/sounio-bootstrap/SKILL.md
check_grep "skill:bootstrap_has_gate_ref"         "sprint56"        skills/sounio-bootstrap/SKILL.md
echo ""

# --- Group 4: sounio-bootstrap reference file ---
echo "--- Group 4: sounio-bootstrap/references/bootstrap-navigation.md ---"
check_file_exists "skill:bootstrap_nav_exists"    skills/sounio-bootstrap/references/bootstrap-navigation.md
check_grep "skill:bootstrap_nav_has_pipeline"     "run_lex_pipeline"    skills/sounio-bootstrap/references/bootstrap-navigation.md
check_grep "skill:bootstrap_nav_has_modules"      "self-hosted/check"   skills/sounio-bootstrap/references/bootstrap-navigation.md
echo ""

# --- Group 5: sounio-pgo skill ---
echo "--- Group 5: sounio-pgo/SKILL.md ---"
check_grep "skill:pgo_has_name"             "^name:"              skills/sounio-pgo/SKILL.md
check_grep "skill:pgo_has_description"      "^description:"       skills/sounio-pgo/SKILL.md
check_grep "skill:pgo_has_workflow"         "## Workflow"         skills/sounio-pgo/SKILL.md
check_grep "skill:pgo_has_pipeline_table"   "opt_cleanup"         skills/sounio-pgo/SKILL.md
check_grep "skill:pgo_has_thresholds"       "1000"                skills/sounio-pgo/SKILL.md
check_grep "skill:pgo_has_gate_ref"         "sprint52"            skills/sounio-pgo/SKILL.md
check_grep "skill:pgo_has_strategy_flow"    "INSTRUMENTED"        skills/sounio-pgo/SKILL.md
echo ""

# --- Summary ---
echo "=== Sprint 59 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"
echo ""

if [ "$FAIL" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
    echo "STATUS: PASS"
    exit 0
else
    echo "STATUS: FAIL"
    exit 1
fi
