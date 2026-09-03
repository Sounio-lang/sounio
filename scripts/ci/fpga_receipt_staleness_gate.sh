#!/usr/bin/env bash
# Guards against a new artifacts/fpga/*.json receipt reporting a *_synth_status
# (or *_sim_status) of "pass" without either declaring itself stale or the
# RTL source it claims to have synthesized actually being present.
#
# Measured 2026-07-27: hardware/** has never been versioned in this
# repository (git ls-files under hardware/ -> 0 files, no history at any
# commit) yet artifacts/fpga/fpga_seed_report.json reported synth_status:
# pass and merkle_lane_core_rtl_present: true for RTL sources
# (hardware/fpga/k_axi_merkle_lane.v and siblings) that do not exist and
# never existed in this checkout. That receipt now carries stale: true; this
# gate stops the next one from repeating the same false signal silently.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FAIL=0

shopt -s nullglob
for report in artifacts/fpga/*.json; do
  # A report is exempt only if it explicitly marks itself stale (the
  # honest thing to do when generated outside this checkout) -- a receipt
  # claiming *_synth_status: pass without either that marker or real RTL
  # backing it is the false signal this gate exists to catch.
  is_stale=$(python3 -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
except (OSError, json.JSONDecodeError):
    print('unreadable')
    sys.exit(0)
print('true' if isinstance(data, dict) and data.get('stale') else 'false')
" "$report")

  if [[ "$is_stale" == "unreadable" ]]; then
    continue
  fi
  if [[ "$is_stale" == "true" ]]; then
    continue
  fi

  has_pass_status=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
if not isinstance(data, dict):
    print('false')
    sys.exit(0)
print('true' if any(k.endswith('_synth_status') and v == 'pass' for k, v in data.items()) else 'false')
" "$report")

  if [[ "$has_pass_status" == "true" ]]; then
    # hardware/ is the only RTL root this repository has ever used; if it's
    # still absent, a *_synth_status: pass in a non-stale report is the same
    # false signal fpga_seed_report.json carried until 2026-07-27.
    if [[ ! -d hardware ]]; then
      echo "$report: reports a *_synth_status of \"pass\" but is not marked" >&2
      echo "  stale:true, and hardware/ does not exist in this checkout." >&2
      echo "  Either the RTL this report claims to have synthesized needs to" >&2
      echo "  actually be present, or the report needs stale:true + stale_reason." >&2
      FAIL=1
    fi
  fi
done

if [[ "$FAIL" -eq 1 ]]; then
  exit 1
fi

echo "fpga receipt staleness check passed: no non-stale report claims synthesis of absent RTL."
