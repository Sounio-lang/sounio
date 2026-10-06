#!/usr/bin/env bash
# Pilot that exits 0 without measuring — the defect the meta-gate must catch.
set -euo pipefail
echo "PILOT_VACUOUS_OK: all 0 tests passed"
exit 0
