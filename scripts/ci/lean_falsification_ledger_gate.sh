#!/usr/bin/env bash
# CI gate for Lean formalization of Falsification Ledger claim logic.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LEAN_FILE="${REPO_ROOT}/formal/lean4/SounioFalsificationLedger.lean"
LEAN_DIR="${REPO_ROOT}/formal/lean4"

fail() {
    echo "LEAN_FALSIFICATION_LEDGER_GATE_FAIL: $*" >&2
    exit 1
}

# F1: Lean file exists
[[ -f "${LEAN_FILE}" ]] || fail "missing ${LEAN_FILE}"

# F1: compiles with lake build
cd "${LEAN_DIR}"
if command -v lake >/dev/null 2>&1; then
    lake build SounioFalsificationLedger || fail "lake build failed"
    echo "F1_LEAN_COMPILES PASS"
else
    echo "SKIP: lake not found; skipping compile check"
fi

# F5: no clinical terms
if grep -Eqi '(patient|diagnosis|treatment|clinical|therapy|disease|symptom)' "${LEAN_FILE}"; then
    fail "clinical term found in Lean file"
fi
echo "F5_NO_CLINICAL_CLAIM PASS"

echo "LEAN_FALSIFICATION_LEDGER_GATE_OK"
