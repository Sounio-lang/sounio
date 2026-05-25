#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DOC="docs/research/moonshot-a-slurm-blocker-handoff.md"
test -f "$DOC"

grep -q 'Blocker-ID: `BLOCKER-MOONSHOT-A-SLURM-RESOURCES`' "$DOC"
grep -q 'Severity: `B3`' "$DOC"
grep -q 'Class: `platform-resource`' "$DOC"
grep -q 'Evidence-Level: `E3`' "$DOC"
grep -q 'SOUNIO_MOONSHOT_A_RUN_SLURM=1 bash scripts/ci/moonshot_a_phase_status_gate.sh' "$DOC"

echo "moonshot_a_slurm_blocker_handoff_gate: PASS doc=$DOC"
