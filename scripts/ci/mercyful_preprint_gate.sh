#!/usr/bin/env bash
# CI gate for the Mercyful Learning preprint (docs/papers/mercyful_learning_preprint_2026-07-26.md).
#
# Checks:
#   1. The preprint exists and carries the required honesty markers
#      (anti-Goodhart, scope statement, falsifiers, GAIDeT-ICMJE disclosure).
#   2. The exact benchmark numbers cited in the preprint still reproduce
#      through the three underlying mercyful gates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PREPRINT="${REPO_ROOT}/docs/papers/mercyful_learning_preprint_2026-07-26.md"

if [[ ! -f "${PREPRINT}" ]]; then
    echo "MERCYFUL_PREPRINT_GATE_FAIL: missing ${PREPRINT}"
    exit 1
fi

REQUIRED_MARKERS=(
    "anti-Goodhart"
    "synthetic"
    "no clinical"
    "Falsifiers"
    "GAIDeT-ICMJE"
    "M_GREEN"
)

for marker in "${REQUIRED_MARKERS[@]}"; do
    if ! grep -qi -- "${marker}" "${PREPRINT}"; then
        echo "MERCYFUL_PREPRINT_GATE_FAIL: preprint missing required marker: ${marker}"
        exit 1
    fi
done

echo "Preprint markers OK; re-running underlying contract gates..."

bash "${REPO_ROOT}/scripts/ci/mercyful_runtime_gate.sh"
bash "${REPO_ROOT}/scripts/ci/mercyful_sounio_gate.sh"
bash "${REPO_ROOT}/scripts/ci/mercyful_clinical_sequencing_gate.sh"

echo "MERCYFUL_PREPRINT_GATE_OK"
