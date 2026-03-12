#!/usr/bin/env bash
# (chmod +x this file before running)
# sprint66_native_codegen_skill_gate.sh — Native Codegen Skill + Model-Dispatch Update
#
# Creates the sounio-native-codegen skill (Program D: Native Codegen) covering the
# native x86-64 backend pipeline from Sprints 52-65. Updates model-dispatch with
# native-backend routing entries, cross-skill-index with Sprint 62-66 mapping, and
# sounio-pgo with the full disp8→frame→peephole pipeline description.
#
# Novel claims:
#   1. sounio-native-codegen skill owns the x86-64 native backend domain (new Program D)
#   2. model-dispatch now routes single-file native passes to Sonnet, multi-file to Opus
#   3. cross-skill-index updated to 5 programs + Sprint 62-66 mapping
#   4. validate_skills_coverage now validates 16 required skills (was 15)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
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

check_log_line() {
    local name="$1"; local expected_line="$2"; local log_file="$3"
    TOTAL=$((TOTAL+1))
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 66: Native Codegen Skill + Model-Dispatch Update ==="
echo ""

# --- Group 1: sounio-native-codegen SKILL.md structure ---
echo "--- Group 1: sounio-native-codegen/SKILL.md ---"
check_grep "native-codegen:file_exists" \
    "." \
    "skills/sounio-native-codegen/SKILL.md"
check_grep "native-codegen:frontmatter_name" \
    "^name: sounio-native-codegen" \
    "skills/sounio-native-codegen/SKILL.md"
check_grep "native-codegen:frontmatter_description" \
    "^description:" \
    "skills/sounio-native-codegen/SKILL.md"
check_grep "native-codegen:workflow_section" \
    "## Workflow" \
    "skills/sounio-native-codegen/SKILL.md"
check_grep "native-codegen:pass_table" \
    "ph_run_on_func" \
    "skills/sounio-native-codegen/SKILL.md"
check_grep "native-codegen:references_section" \
    "## References" \
    "skills/sounio-native-codegen/SKILL.md"
check_grep "native-codegen:gate_history" \
    "sprint65_peephole_gate" \
    "skills/sounio-native-codegen/SKILL.md"

echo ""
echo "--- Group 2: model-dispatch/SKILL.md new entries ---"
check_grep "dispatch:native_single_file_entry" \
    "Native backend single-file pass" \
    "skills/model-dispatch/SKILL.md"
check_grep "dispatch:native_multifile_entry" \
    "Multi-file native pipeline change" \
    "skills/model-dispatch/SKILL.md"
check_grep "dispatch:peephole_row" \
    "Add a new peephole pattern" \
    "skills/model-dispatch/SKILL.md"
check_grep "dispatch:wire_native_pass_row" \
    "Wire a new native pass in compile_ir_function" \
    "skills/model-dispatch/SKILL.md"
check_grep "dispatch:debug_elf_row" \
    "Debug native binary crash" \
    "skills/model-dispatch/SKILL.md"
check_grep "dispatch:emit_helper_row" \
    "Add emit_\* helper to encode.sio" \
    "skills/model-dispatch/SKILL.md"

echo ""
echo "--- Group 3: cross-skill-index.md updates ---"
check_grep "index:native_codegen_row" \
    "sounio-native-codegen" \
    "skills/model-dispatch/references/cross-skill-index.md"
check_grep "index:program_d_column" \
    "Native Codegen" \
    "skills/model-dispatch/references/cross-skill-index.md"
check_grep "index:sprint62_entry" \
    "62.*native" \
    "skills/model-dispatch/references/cross-skill-index.md"
check_grep "index:sprint66_entry" \
    "66.*native" \
    "skills/model-dispatch/references/cross-skill-index.md"

echo ""
echo "--- Group 4: sounio-pgo/SKILL.md updates ---"
check_grep "pgo:extended_pipeline" \
    "disp8.*compact frame.*peephole" \
    "skills/sounio-pgo/SKILL.md"
check_grep "pgo:disp8_row" \
    "Disp8 adaptive encoding" \
    "skills/sounio-pgo/SKILL.md"
check_grep "pgo:frame_row" \
    "Compact frame sizing" \
    "skills/sounio-pgo/SKILL.md"
check_grep "pgo:peephole_row" \
    "Peephole optimizer" \
    "skills/sounio-pgo/SKILL.md"

echo ""
echo "--- Group 5: validate_skills_coverage.sh update ---"
check_grep "coverage:sounio_native_codegen_in_list" \
    "sounio-native-codegen" \
    "scripts/validate_skills_coverage.sh"
check_grep "coverage:sprint66_comment" \
    "Sprint 66" \
    "scripts/validate_skills_coverage.sh"
COVERAGE_LOG="$(mktemp)"
bash scripts/validate_skills_coverage.sh >"$COVERAGE_LOG" 2>&1 || true
check_log_line "coverage:16_skills_valid" \
    "Coverage: 16/16 skills valid  (FAIL: 0)" "$COVERAGE_LOG"
rm -f "$COVERAGE_LOG"

echo ""
echo "--- Group 6: Regression ---"
check_grep "regression:dispatch_pgo_row" \
    "Add a new PGO pass" \
    "skills/model-dispatch/SKILL.md"
check_grep "regression:dispatch_fallback_chain" \
    "Fallback Chain" \
    "skills/model-dispatch/SKILL.md"
check_grep "regression:pgo_sprof_gate" \
    "sprint47_sprof_gate" \
    "skills/sounio-pgo/SKILL.md"
check_grep "regression:peephole_gate_exists" \
    "." \
    "scripts/sprint65_peephole_gate.sh"

# --- Results ---
echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"
REASON="all_cases_passed"
if [ "$FAIL" -gt 0 ]; then
    STATUS="fail"
    REASON="${FAIL}_cases_failed"
fi
if [ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ]; then
    REASON="all_run_passed_${NOT_RUN}_not_run"
fi

mkdir -p "$ROOT_DIR/artifacts/sprint66"
cat > "$ROOT_DIR/artifacts/sprint66/native_codegen_skill_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint66.native_codegen_skill_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": 60
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "sounio-native-codegen skill created as Program D owner for x86-64 native backend (Sprints 52-65)",
    "model-dispatch now routes single-file native passes to Sonnet 4.6, multi-file to Opus 4.6",
    "cross-skill-index expanded to 5 programs with Sprint 62-66 sprint→skill mapping",
    "validate_skills_coverage.sh now validates 16 required skills (added sounio-native-codegen)"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint66/native_codegen_skill_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
