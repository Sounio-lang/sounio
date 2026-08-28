#!/usr/bin/env bash
# Pilot gate that measures something and emits a receipt. Used only by
# gate_measurement_meta_gate.sh — not a production CI entrypoint.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=scripts/lib/gate_assert.sh
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
# shellcheck source=scripts/lib/gate_measurement_receipt.sh
. "$ROOT_DIR/scripts/lib/gate_measurement_receipt.sh"
gate_name "gate_measurement_pilot_ok"
gate_measurement_reset
# Three synthetic assertions (stand-ins for real checks).
gate_measurement_add 1
gate_measurement_add 1
gate_measurement_add 1
gate_measurement_emit pass
gate_pass "pilot measured 3"
