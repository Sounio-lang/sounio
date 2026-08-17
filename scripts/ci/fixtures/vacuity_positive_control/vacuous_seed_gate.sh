#!/usr/bin/env bash
# DELIBERATE VACUOUS GATE — positive control for gate_vacuous_fixture_sweep.sh
#
# GATE_CONTRACT: v0
# GATE_ID: vacuous_seed_positive_control
# GATE_CLAIMS: must be flagged as cases=0 by the mechanical vacuity sweep
# GATE_ENGINE: n/a
# GATE_RESULT_ON_SKIP: forbidden
#
# This gate is NOT for CI wiring. It exists so the sweep instrument can prove
# it returns non-zero findings. Do not "fix" it by adding fixtures.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
# Glob over an empty directory — nullglob makes the loop body never run.
shopt -s nullglob
count=0
for f in "$ROOT"/scripts/ci/fixtures/vacuity_positive_control/*.sio; do
  count=$((count + 1))
  echo "would check $f"
done
echo "VACUOUS_SEED_GATE_OK cases=$count"
# Exit 0 even when cases=0 — that is the unmeasure shape under test.
exit 0
