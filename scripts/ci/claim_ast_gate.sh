#!/usr/bin/env bash
# CI gate for AST-native claims (preprocessor path).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLAIM_SIO="${REPO_ROOT}/stdlib/epistemic/claim_ast.sio"
PREPROCESSOR="${REPO_ROOT}/scripts/research/claim_ast_preprocessor.py"

fail() {
    echo "CLAIM_AST_GATE_FAIL: $*" >&2
    exit 1
}

# A1: struct defined and type-checks
[[ -f "${CLAIM_SIO}" ]] || fail "missing ${CLAIM_SIO}"
"${REPO_ROOT}/bin/souc" check "${CLAIM_SIO}" > /dev/null 2>&1 || fail "claim_ast.sio does not type-check"
echo "A1_STRUCT_DEFINED PASS"

# A2: preprocessor identity on empty input
RESULT="$(echo '' | python3 "${PREPROCESSOR}")"
[[ -z "${RESULT}" ]] || fail "preprocessor modified empty input"
echo "A2_PREPROCESSOR_IDENTITY PASS"

# A3/A4: preprocessor roundtrip
python3 "${PREPROCESSOR}" --test || fail "preprocessor roundtrip failed"
echo "A3_A4_PREPROCESSOR_ROUNDTRIP PASS"

# A5: no parser files modified in this diff. Capture first: through the pipe
# a failing `git diff` reads as "no parser files" (grep's empty result
# decides), which is a silent pass on a broken instrument; the capture keeps
# git's own status visible.
changed_paths="$(git diff --name-only HEAD)"
if grep -q '^self-hosted/parser/' <<<"$changed_paths"; then
    fail "parser files modified"
fi
echo "A5_NO_PARSER_TOUCH PASS"

echo "CLAIM_AST_GATE_OK"
