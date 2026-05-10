#!/usr/bin/env bash
# validate_skills_coverage.sh -- Validate all required skills exist with correct structure
#
# Every active skill must have:
#   - skills/<name>/SKILL.md
#   - YAML frontmatter: "name:" and "description:" fields
#   - "## Workflow" section
#
# Exit 0 if all pass, exit 1 if any fail.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; TOTAL=0

# Full list of required skills (12 original + 3 added in Sprint 59 + 1 added in Sprint 66)
REQUIRED_SKILLS=(
    "sounio-language"
    "sounio-l0-core"
    "sounio-tests-suite"
    "sounio-codegen-backends"
    "sounio-typeck-effects"
    "sounio-epistemic-types"
    "sounio-stdlib-dev"
    "sounio-ontology"
    "sounio-compiler-mir"
    "sounio-examples-hygiene"
    "sounio-tooling"
    "model-dispatch"
    "sounio-render"
    "sounio-bootstrap"
    "sounio-pgo"
    "sounio-native-codegen"
)

for skill in "${REQUIRED_SKILLS[@]}"; do
    local_path="skills/$skill/SKILL.md"
    TOTAL=$((TOTAL+1))

    if [ ! -f "$local_path" ]; then
        echo "FAIL  coverage:$skill (SKILL.md missing at $local_path)"
        FAIL=$((FAIL+1))
        continue
    fi

    if ! grep -q "^name:" "$local_path"; then
        echo "FAIL  coverage:$skill (missing 'name:' frontmatter field)"
        FAIL=$((FAIL+1))
        continue
    fi

    if ! grep -q "^description:" "$local_path"; then
        echo "FAIL  coverage:$skill (missing 'description:' frontmatter field)"
        FAIL=$((FAIL+1))
        continue
    fi

    if ! grep -q "## Workflow" "$local_path"; then
        echo "FAIL  coverage:$skill (missing '## Workflow' section)"
        FAIL=$((FAIL+1))
        continue
    fi

    echo "PASS  coverage:$skill"
    PASS=$((PASS+1))
done

echo ""
echo "Coverage: $PASS/$TOTAL skills valid  (FAIL: $FAIL)"

if [ "$FAIL" -eq 0 ]; then
    exit 0
else
    exit 1
fi
