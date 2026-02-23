#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p artifacts/fpga

REQUIRE_FPGA_PASS="${OMEGA_REQUIRE_FPGA_PASS:-0}"
REQUIRE_QUANTUM_CONTROLLER="${OMEGA_REQUIRE_QUANTUM_CONTROLLER:-0}"
SIM_STATUS="skipped"
SYNTH_STATUS="skipped"
SIM_RC=0
SYNTH_RC=0
QCTRL_SIM_STATUS="skipped"
QCTRL_SIM_RC=0
QCTRL_SYNTH_STATUS="skipped"
QCTRL_SYNTH_RC=0

echo "==> FPGA seed simulation"
if command -v iverilog >/dev/null 2>&1; then
  set +e
  iverilog -o artifacts/fpga/tb_epistemic_mac hardware/fpga/epistemic_mac.v hardware/fpga/tb_epistemic_mac.v >artifacts/fpga/iverilog.log 2>&1
  SIM_RC=$?
  if [ "$SIM_RC" -eq 0 ]; then
    vvp artifacts/fpga/tb_epistemic_mac >artifacts/fpga/vvp.log 2>&1
    SIM_RC=$?
  fi
  set -e
  if [ "$SIM_RC" -eq 0 ]; then
    SIM_STATUS="pass"
  else
    SIM_STATUS="fail"
    echo "warning: FPGA simulation failed (rc=$SIM_RC); see artifacts/fpga/vvp.log"
  fi
else
  echo "warning: iverilog not found; skipping simulation"
  if [ "$REQUIRE_FPGA_PASS" = "1" ]; then
    SIM_STATUS="missing_tool"
    SIM_RC=127
  fi
fi

echo "==> FPGA quantum controller simulation"
if [ -f hardware/fpga/epistemic_quantum_controller.v ] && [ -f hardware/fpga/tb_epistemic_quantum_controller.v ]; then
  if command -v iverilog >/dev/null 2>&1; then
    set +e
    iverilog -o artifacts/fpga/tb_epistemic_quantum_controller \
      hardware/fpga/epistemic_quantum_controller.v \
      hardware/fpga/tb_epistemic_quantum_controller.v >artifacts/fpga/iverilog_quantum_controller.log 2>&1
    QCTRL_SIM_RC=$?
    if [ "$QCTRL_SIM_RC" -eq 0 ]; then
      vvp artifacts/fpga/tb_epistemic_quantum_controller >artifacts/fpga/vvp_quantum_controller.log 2>&1
      QCTRL_SIM_RC=$?
    fi
    set -e
    if [ "$QCTRL_SIM_RC" -eq 0 ]; then
      QCTRL_SIM_STATUS="pass"
    else
      QCTRL_SIM_STATUS="fail"
      echo "warning: quantum controller simulation failed (rc=$QCTRL_SIM_RC); see artifacts/fpga/vvp_quantum_controller.log"
    fi
  else
    echo "warning: iverilog not found; skipping quantum controller simulation"
    if [ "$REQUIRE_QUANTUM_CONTROLLER" = "1" ]; then
      QCTRL_SIM_STATUS="missing_tool"
      QCTRL_SIM_RC=127
    fi
  fi
else
  echo "warning: quantum controller sources not found; skipping"
  if [ "$REQUIRE_QUANTUM_CONTROLLER" = "1" ]; then
    QCTRL_SIM_STATUS="missing_sources"
    QCTRL_SIM_RC=2
  fi
fi

echo "==> FPGA seed synthesis"
if command -v yosys >/dev/null 2>&1; then
  set +e
  yosys -s hardware/fpga/synth.ys >artifacts/fpga/yosys.log 2>&1
  SYNTH_RC=$?
  set -e
  if [ "$SYNTH_RC" -eq 0 ]; then
    SYNTH_STATUS="pass"
  else
    SYNTH_STATUS="fail"
    echo "warning: FPGA synthesis failed (rc=$SYNTH_RC); see artifacts/fpga/yosys.log"
  fi
else
  echo "warning: yosys not found; skipping synthesis"
  if [ "$REQUIRE_FPGA_PASS" = "1" ]; then
    SYNTH_STATUS="missing_tool"
    SYNTH_RC=127
  fi
fi

echo "==> FPGA quantum controller synthesis"
if [ -f hardware/fpga/synth_quantum_controller.ys ]; then
  if command -v yosys >/dev/null 2>&1; then
    set +e
    yosys -s hardware/fpga/synth_quantum_controller.ys >artifacts/fpga/yosys_quantum_controller.log 2>&1
    QCTRL_SYNTH_RC=$?
    set -e
    if [ "$QCTRL_SYNTH_RC" -eq 0 ]; then
      QCTRL_SYNTH_STATUS="pass"
    else
      QCTRL_SYNTH_STATUS="fail"
      echo "warning: quantum controller synthesis failed (rc=$QCTRL_SYNTH_RC); see artifacts/fpga/yosys_quantum_controller.log"
    fi
  else
    echo "warning: yosys not found; skipping quantum controller synthesis"
    if [ "$REQUIRE_QUANTUM_CONTROLLER" = "1" ]; then
      QCTRL_SYNTH_STATUS="missing_tool"
      QCTRL_SYNTH_RC=127
    fi
  fi
else
  echo "warning: hardware/fpga/synth_quantum_controller.ys not found; skipping"
  if [ "$REQUIRE_QUANTUM_CONTROLLER" = "1" ]; then
    QCTRL_SYNTH_STATUS="missing_sources"
    QCTRL_SYNTH_RC=2
  fi
fi

SIM_STATUS="$SIM_STATUS" \
SIM_RC="$SIM_RC" \
SYNTH_STATUS="$SYNTH_STATUS" \
SYNTH_RC="$SYNTH_RC" \
QCTRL_SIM_STATUS="$QCTRL_SIM_STATUS" \
QCTRL_SIM_RC="$QCTRL_SIM_RC" \
QCTRL_SYNTH_STATUS="$QCTRL_SYNTH_STATUS" \
QCTRL_SYNTH_RC="$QCTRL_SYNTH_RC" \
python3 - <<'PY'
import json
import os
from pathlib import Path

report = {
    "schema": "sounio.omega.fpga-seed-report.v1",
    "sim_status": os.environ["SIM_STATUS"],
    "sim_rc": int(os.environ["SIM_RC"]),
    "synth_status": os.environ["SYNTH_STATUS"],
    "synth_rc": int(os.environ["SYNTH_RC"]),
    "quantum_controller_sim_status": os.environ["QCTRL_SIM_STATUS"],
    "quantum_controller_sim_rc": int(os.environ["QCTRL_SIM_RC"]),
    "quantum_controller_synth_status": os.environ["QCTRL_SYNTH_STATUS"],
    "quantum_controller_synth_rc": int(os.environ["QCTRL_SYNTH_RC"]),
}
Path("artifacts/fpga/fpga_seed_report.json").write_text(json.dumps(report, indent=2))
print("fpga_seed_report:", report)
PY

if [ "$REQUIRE_FPGA_PASS" = "1" ]; then
  if [ "$SIM_STATUS" != "pass" ] || [ "$SYNTH_STATUS" != "pass" ]; then
    echo "error: FPGA seed strict gate failed (sim=$SIM_STATUS synth=$SYNTH_STATUS)" >&2
    exit 2
  fi
fi

if [ "$REQUIRE_QUANTUM_CONTROLLER" = "1" ]; then
  if [ "$QCTRL_SIM_STATUS" != "pass" ] || [ "$QCTRL_SYNTH_STATUS" != "pass" ]; then
    echo "error: quantum controller strict gate failed (sim=$QCTRL_SIM_STATUS synth=$QCTRL_SYNTH_STATUS)" >&2
    exit 2
  fi
fi

echo "FPGA_EPISTEMIC_SEED_DONE"
