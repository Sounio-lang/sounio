#!/usr/bin/env bash
# Mechanical vacuous-fixture sweep. See gate_vacuous_fixture_sweep.py.
# Exit 2 if the planted positive-control seed is not flagged VACUOUS.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="${GATE_VACUOUS_SWEEP_DIR:-$(mktemp -d /tmp/sounio-vacuous-fixture-sweep.XXXXXX)}"
export ROOT_DIR OUT_DIR
python3 "$ROOT_DIR/scripts/ci/gate_vacuous_fixture_sweep.py"
# shellcheck disable=SC1091
source "$OUT_DIR/status.env"
echo "out_dir=$OUT_DIR"
if [[ "${SEED_FIRED}" != "1" ]]; then
  echo "GATE_VACUOUS_FIXTURE_SWEEP_INSTRUMENT_FAIL: positive control did not fire" >&2
  exit 2
fi
echo "GATE_VACUOUS_FIXTURE_SWEEP_OK scanned=$SCANNED vacuous=$VACUOUS seed_fired=1"
exit 0
