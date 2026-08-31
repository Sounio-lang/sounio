#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p artifacts/fpga
WAVE_DIR="artifacts/fpga/waveforms"
mkdir -p "$WAVE_DIR"

REQUIRE_FPGA_PASS="${OMEGA_REQUIRE_FPGA_PASS:-0}"
REQUIRE_QUANTUM_CONTROLLER="${OMEGA_REQUIRE_QUANTUM_CONTROLLER:-0}"
REQUIRE_QUANTUM_ACCUM_LINK="${OMEGA_REQUIRE_QUANTUM_ACCUM_LINK:-0}"
REQUIRE_K_AXI="${OMEGA_REQUIRE_K_AXI:-0}"
REQUIRE_K_AXI_RETURN="${OMEGA_REQUIRE_K_AXI_RETURN:-0}"
REQUIRE_EPI_POWER_ACCUM="${OMEGA_REQUIRE_EPI_POWER_ACCUM:-0}"
REQUIRE_MERKLE_LANE="${OMEGA_REQUIRE_MERKLE_LANE:-0}"
REQUIRE_MERKLE_ROOT="${OMEGA_REQUIRE_MERKLE_ROOT:-0}"
REQUIRE_COUNTER_REPRO="${OMEGA_REQUIRE_COUNTER_REPRO:-0}"
REQUIRE_RESOURCE_TREND="${OMEGA_REQUIRE_RESOURCE_TREND:-0}"
SIM_STATUS="skipped"
SYNTH_STATUS="skipped"
SIM_RC=0
SYNTH_RC=0
QCTRL_SIM_STATUS="skipped"
QCTRL_SIM_RC=0
QCTRL_SYNTH_STATUS="skipped"
QCTRL_SYNTH_RC=0
QLINK_SIM_STATUS="skipped"
QLINK_SIM_RC=0
KAXI_SIM_STATUS="skipped"
KAXI_SIM_RC=0
KAXI_SYNTH_STATUS="skipped"
KAXI_SYNTH_RC=0
KAXI_RETURN_SIM_STATUS="skipped"
KAXI_RETURN_SIM_RC=0
KAXI_RETURN_SYNTH_STATUS="skipped"
KAXI_RETURN_SYNTH_RC=0
EPI_PWR_SIM_STATUS="skipped"
EPI_PWR_SIM_RC=0
EPI_PWR_SYNTH_STATUS="skipped"
EPI_PWR_SYNTH_RC=0
MERKLE_SYNTH_STATUS="skipped"
MERKLE_SYNTH_RC=0
MERKLE_ROOT_SYNTH_STATUS="skipped"
MERKLE_ROOT_SYNTH_RC=0

echo "==> FPGA seed simulation"
if command -v iverilog >/dev/null 2>&1; then
  set +e
  iverilog -o artifacts/fpga/tb_epistemic_mac hardware/fpga/epistemic_mac.v hardware/fpga/tb_epistemic_mac.v >artifacts/fpga/iverilog.log 2>&1
  SIM_RC=$?
  if [ "$SIM_RC" -eq 0 ]; then
    vvp artifacts/fpga/tb_epistemic_mac +dumpfile="$WAVE_DIR/tb_epistemic_mac.vcd" >artifacts/fpga/vvp.log 2>&1
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
      vvp artifacts/fpga/tb_epistemic_quantum_controller +dumpfile="$WAVE_DIR/tb_epistemic_quantum_controller.vcd" >artifacts/fpga/vvp_quantum_controller.log 2>&1
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

echo "==> FPGA quantum->accumulator link simulation"
if [ -f hardware/fpga/epistemic_quantum_controller.v ] \
  && [ -f hardware/fpga/epistemic_power_accumulator.v ] \
  && [ -f hardware/fpga/tb_quantum_accumulator_link.v ]; then
  if command -v iverilog >/dev/null 2>&1; then
    set +e
    iverilog -o artifacts/fpga/tb_quantum_accumulator_link \
      hardware/fpga/epistemic_quantum_controller.v \
      hardware/fpga/epistemic_power_accumulator.v \
      hardware/fpga/tb_quantum_accumulator_link.v >artifacts/fpga/iverilog_quantum_accumulator_link.log 2>&1
    QLINK_SIM_RC=$?
    if [ "$QLINK_SIM_RC" -eq 0 ]; then
      vvp artifacts/fpga/tb_quantum_accumulator_link +dumpfile="$WAVE_DIR/tb_quantum_accumulator_link.vcd" >artifacts/fpga/vvp_quantum_accumulator_link.log 2>&1
      QLINK_SIM_RC=$?
    fi
    set -e
    if [ "$QLINK_SIM_RC" -eq 0 ]; then
      QLINK_SIM_STATUS="pass"
    else
      QLINK_SIM_STATUS="fail"
      echo "warning: quantum->accumulator link simulation failed (rc=$QLINK_SIM_RC); see artifacts/fpga/vvp_quantum_accumulator_link.log"
    fi
  else
    echo "warning: iverilog not found; skipping quantum->accumulator link simulation"
    if [ "$REQUIRE_QUANTUM_ACCUM_LINK" = "1" ]; then
      QLINK_SIM_STATUS="missing_tool"
      QLINK_SIM_RC=127
    fi
  fi
else
  echo "warning: quantum->accumulator link sources not found; skipping"
  if [ "$REQUIRE_QUANTUM_ACCUM_LINK" = "1" ]; then
    QLINK_SIM_STATUS="missing_sources"
    QLINK_SIM_RC=2
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

echo "==> K-AXI bridge simulation"
if [ -f hardware/fpga/k_axi_master.v ] && [ -f hardware/fpga/k_axi_slave_epistemic.v ] && [ -f hardware/fpga/tb_k_axi_bridge.v ]; then
  if command -v iverilog >/dev/null 2>&1; then
    set +e
    iverilog -o artifacts/fpga/tb_k_axi_bridge \
      hardware/fpga/k_axi_master.v \
      hardware/fpga/k_axi_slave_epistemic.v \
      hardware/fpga/tb_k_axi_bridge.v >artifacts/fpga/iverilog_k_axi.log 2>&1
    KAXI_SIM_RC=$?
    if [ "$KAXI_SIM_RC" -eq 0 ]; then
      vvp artifacts/fpga/tb_k_axi_bridge +dumpfile="$WAVE_DIR/tb_k_axi_bridge.vcd" >artifacts/fpga/vvp_k_axi.log 2>&1
      KAXI_SIM_RC=$?
    fi
    set -e
    if [ "$KAXI_SIM_RC" -eq 0 ]; then
      KAXI_SIM_STATUS="pass"
    else
      KAXI_SIM_STATUS="fail"
      echo "warning: K-AXI simulation failed (rc=$KAXI_SIM_RC); see artifacts/fpga/vvp_k_axi.log"
    fi
  else
    echo "warning: iverilog not found; skipping K-AXI simulation"
    if [ "$REQUIRE_K_AXI" = "1" ]; then
      KAXI_SIM_STATUS="missing_tool"
      KAXI_SIM_RC=127
    fi
  fi
else
  echo "warning: K-AXI sources not found; skipping"
  if [ "$REQUIRE_K_AXI" = "1" ]; then
    KAXI_SIM_STATUS="missing_sources"
    KAXI_SIM_RC=2
  fi
fi

echo "==> K-AXI bridge synthesis"
if [ -f hardware/fpga/synth_k_axi.ys ]; then
  if command -v yosys >/dev/null 2>&1; then
    set +e
    yosys -s hardware/fpga/synth_k_axi.ys >artifacts/fpga/yosys_k_axi.log 2>&1
    KAXI_SYNTH_RC=$?
    set -e
    if [ "$KAXI_SYNTH_RC" -eq 0 ]; then
      KAXI_SYNTH_STATUS="pass"
    else
      KAXI_SYNTH_STATUS="fail"
      echo "warning: K-AXI synthesis failed (rc=$KAXI_SYNTH_RC); see artifacts/fpga/yosys_k_axi.log"
    fi
  else
    echo "warning: yosys not found; skipping K-AXI synthesis"
    if [ "$REQUIRE_K_AXI" = "1" ]; then
      KAXI_SYNTH_STATUS="missing_tool"
      KAXI_SYNTH_RC=127
    fi
  fi
else
  echo "warning: hardware/fpga/synth_k_axi.ys not found; skipping"
  if [ "$REQUIRE_K_AXI" = "1" ]; then
    KAXI_SYNTH_STATUS="missing_sources"
    KAXI_SYNTH_RC=2
  fi
fi

echo "==> K-AXI Merkle lane synthesis"
if [ -f hardware/fpga/k_axi_merkle_lane.v ]; then
  if command -v yosys >/dev/null 2>&1; then
    set +e
    yosys -p "read_verilog hardware/fpga/k_axi_merkle_lane.v; synth -top k_axi_merkle_lane_core" >artifacts/fpga/yosys_k_axi_merkle_lane.log 2>&1
    MERKLE_SYNTH_RC=$?
    set -e
    if [ "$MERKLE_SYNTH_RC" -eq 0 ]; then
      MERKLE_SYNTH_STATUS="pass"
    else
      MERKLE_SYNTH_STATUS="fail"
      echo "warning: K-AXI Merkle lane synthesis failed (rc=$MERKLE_SYNTH_RC); see artifacts/fpga/yosys_k_axi_merkle_lane.log"
    fi
  else
    echo "warning: yosys not found; skipping K-AXI Merkle lane synthesis"
    if [ "$REQUIRE_MERKLE_LANE" = "1" ]; then
      MERKLE_SYNTH_STATUS="missing_tool"
      MERKLE_SYNTH_RC=127
    fi
  fi
else
  echo "warning: hardware/fpga/k_axi_merkle_lane.v not found; skipping"
  if [ "$REQUIRE_MERKLE_LANE" = "1" ]; then
    MERKLE_SYNTH_STATUS="missing_sources"
    MERKLE_SYNTH_RC=2
  fi
fi

echo "==> K-AXI Merkle root synthesis"
if [ -f hardware/fpga/k_axi_merkle_root_lane.v ]; then
  if command -v yosys >/dev/null 2>&1; then
    set +e
    yosys -p "read_verilog hardware/fpga/k_axi_merkle_root_lane.v; synth -top k_axi_merkle_root_lane_core" >artifacts/fpga/yosys_k_axi_merkle_root_lane.log 2>&1
    MERKLE_ROOT_SYNTH_RC=$?
    set -e
    if [ "$MERKLE_ROOT_SYNTH_RC" -eq 0 ]; then
      MERKLE_ROOT_SYNTH_STATUS="pass"
    else
      MERKLE_ROOT_SYNTH_STATUS="fail"
      echo "warning: K-AXI Merkle root synthesis failed (rc=$MERKLE_ROOT_SYNTH_RC); see artifacts/fpga/yosys_k_axi_merkle_root_lane.log"
    fi
  else
    echo "warning: yosys not found; skipping K-AXI Merkle root synthesis"
    if [ "$REQUIRE_MERKLE_ROOT" = "1" ]; then
      MERKLE_ROOT_SYNTH_STATUS="missing_tool"
      MERKLE_ROOT_SYNTH_RC=127
    fi
  fi
else
  echo "warning: hardware/fpga/k_axi_merkle_root_lane.v not found; skipping"
  if [ "$REQUIRE_MERKLE_ROOT" = "1" ]; then
    MERKLE_ROOT_SYNTH_STATUS="missing_sources"
    MERKLE_ROOT_SYNTH_RC=2
  fi
fi

echo "==> K-AXI return channel simulation"
if [ -f hardware/fpga/k_axi_return_fifo.v ] \
  && [ -f hardware/fpga/k_axi_return_mux.v ] \
  && [ -f hardware/fpga/tb_k_axi_bidirectional.v ]; then
  if command -v iverilog >/dev/null 2>&1; then
    set +e
    iverilog -o artifacts/fpga/tb_k_axi_bidirectional \
      hardware/fpga/k_axi_return_fifo.v \
      hardware/fpga/k_axi_return_mux.v \
      hardware/fpga/tb_k_axi_bidirectional.v >artifacts/fpga/iverilog_k_axi_return.log 2>&1
    KAXI_RETURN_SIM_RC=$?
    if [ "$KAXI_RETURN_SIM_RC" -eq 0 ]; then
      vvp artifacts/fpga/tb_k_axi_bidirectional +dumpfile="$WAVE_DIR/tb_k_axi_bidirectional.vcd" >artifacts/fpga/vvp_k_axi_return.log 2>&1
      KAXI_RETURN_SIM_RC=$?
    fi
    set -e
    if [ "$KAXI_RETURN_SIM_RC" -eq 0 ]; then
      KAXI_RETURN_SIM_STATUS="pass"
    else
      KAXI_RETURN_SIM_STATUS="fail"
      echo "warning: K-AXI return simulation failed (rc=$KAXI_RETURN_SIM_RC); see artifacts/fpga/vvp_k_axi_return.log"
    fi
  else
    echo "warning: iverilog not found; skipping K-AXI return simulation"
    if [ "$REQUIRE_K_AXI_RETURN" = "1" ]; then
      KAXI_RETURN_SIM_STATUS="missing_tool"
      KAXI_RETURN_SIM_RC=127
    fi
  fi
else
  echo "warning: K-AXI return sources not found; skipping"
  if [ "$REQUIRE_K_AXI_RETURN" = "1" ]; then
    KAXI_RETURN_SIM_STATUS="missing_sources"
    KAXI_RETURN_SIM_RC=2
  fi
fi

echo "==> K-AXI return channel synthesis"
if [ -f hardware/fpga/synth_k_axi_return.ys ]; then
  if command -v yosys >/dev/null 2>&1; then
    set +e
    yosys -s hardware/fpga/synth_k_axi_return.ys >artifacts/fpga/yosys_k_axi_return.log 2>&1
    KAXI_RETURN_SYNTH_RC=$?
    set -e
    if [ "$KAXI_RETURN_SYNTH_RC" -eq 0 ]; then
      KAXI_RETURN_SYNTH_STATUS="pass"
    else
      KAXI_RETURN_SYNTH_STATUS="fail"
      echo "warning: K-AXI return synthesis failed (rc=$KAXI_RETURN_SYNTH_RC); see artifacts/fpga/yosys_k_axi_return.log"
    fi
  else
    echo "warning: yosys not found; skipping K-AXI return synthesis"
    if [ "$REQUIRE_K_AXI_RETURN" = "1" ]; then
      KAXI_RETURN_SYNTH_STATUS="missing_tool"
      KAXI_RETURN_SYNTH_RC=127
    fi
  fi
else
  echo "warning: hardware/fpga/synth_k_axi_return.ys not found; skipping"
  if [ "$REQUIRE_K_AXI_RETURN" = "1" ]; then
    KAXI_RETURN_SYNTH_STATUS="missing_sources"
    KAXI_RETURN_SYNTH_RC=2
  fi
fi

echo "==> Epistemic power accumulator simulation"
if [ -f hardware/fpga/epistemic_power_accumulator.v ] && [ -f hardware/fpga/tb_epistemic_power_accumulator.v ]; then
  if command -v iverilog >/dev/null 2>&1; then
    set +e
    iverilog -o artifacts/fpga/tb_epistemic_power_accumulator \
      hardware/fpga/epistemic_power_accumulator.v \
      hardware/fpga/tb_epistemic_power_accumulator.v >artifacts/fpga/iverilog_epistemic_power.log 2>&1
    EPI_PWR_SIM_RC=$?
    if [ "$EPI_PWR_SIM_RC" -eq 0 ]; then
      vvp artifacts/fpga/tb_epistemic_power_accumulator +dumpfile="$WAVE_DIR/tb_epistemic_power_accumulator.vcd" >artifacts/fpga/vvp_epistemic_power.log 2>&1
      EPI_PWR_SIM_RC=$?
    fi
    set -e
    if [ "$EPI_PWR_SIM_RC" -eq 0 ]; then
      EPI_PWR_SIM_STATUS="pass"
    else
      EPI_PWR_SIM_STATUS="fail"
      echo "warning: epistemic power accumulator simulation failed (rc=$EPI_PWR_SIM_RC); see artifacts/fpga/vvp_epistemic_power.log"
    fi
  else
    echo "warning: iverilog not found; skipping epistemic power accumulator simulation"
    if [ "$REQUIRE_EPI_POWER_ACCUM" = "1" ]; then
      EPI_PWR_SIM_STATUS="missing_tool"
      EPI_PWR_SIM_RC=127
    fi
  fi
else
  echo "warning: epistemic power accumulator sources not found; skipping"
  if [ "$REQUIRE_EPI_POWER_ACCUM" = "1" ]; then
    EPI_PWR_SIM_STATUS="missing_sources"
    EPI_PWR_SIM_RC=2
  fi
fi

echo "==> Epistemic power accumulator synthesis"
if [ -f hardware/fpga/synth_epistemic_power_accumulator.ys ]; then
  if command -v yosys >/dev/null 2>&1; then
    set +e
    yosys -s hardware/fpga/synth_epistemic_power_accumulator.ys >artifacts/fpga/yosys_epistemic_power.log 2>&1
    EPI_PWR_SYNTH_RC=$?
    set -e
    if [ "$EPI_PWR_SYNTH_RC" -eq 0 ]; then
      EPI_PWR_SYNTH_STATUS="pass"
    else
      EPI_PWR_SYNTH_STATUS="fail"
      echo "warning: epistemic power accumulator synthesis failed (rc=$EPI_PWR_SYNTH_RC); see artifacts/fpga/yosys_epistemic_power.log"
    fi
  else
    echo "warning: yosys not found; skipping epistemic power accumulator synthesis"
    if [ "$REQUIRE_EPI_POWER_ACCUM" = "1" ]; then
      EPI_PWR_SYNTH_STATUS="missing_tool"
      EPI_PWR_SYNTH_RC=127
    fi
  fi
else
  echo "warning: hardware/fpga/synth_epistemic_power_accumulator.ys not found; skipping"
  if [ "$REQUIRE_EPI_POWER_ACCUM" = "1" ]; then
    EPI_PWR_SYNTH_STATUS="missing_sources"
    EPI_PWR_SYNTH_RC=2
  fi
fi

MERKLE_LANE_PRESENT=false
K_AXI_RTL_PATH="hardware/fpga/k_axi_slave_epistemic.v"
if [ -f "$K_AXI_RTL_PATH" ]; then
  if grep -q 'merkle_lane_valid' "$K_AXI_RTL_PATH" \
    && grep -q 'merkle_lane_digest' "$K_AXI_RTL_PATH" \
    && grep -q 'merkle_lane_req' "$K_AXI_RTL_PATH" \
    && grep -q 'merkle_lane_seed' "$K_AXI_RTL_PATH"; then
    MERKLE_LANE_PRESENT=true
  fi
fi

MERKLE_LANE_CORE_RTL_PRESENT=false
if [ -f "hardware/fpga/k_axi_merkle_lane.v" ]; then
  if grep -q 'module k_axi_merkle_lane_core' "hardware/fpga/k_axi_merkle_lane.v" \
    && grep -q 'merkle_lane_digest' "hardware/fpga/k_axi_merkle_lane.v" \
    && grep -q 'MERKLE_SALT' "hardware/fpga/k_axi_merkle_lane.v"; then
    MERKLE_LANE_CORE_RTL_PRESENT=true
  fi
fi

MERKLE_ROOT_CORE_RTL_PRESENT=false
if [ -f "hardware/fpga/k_axi_merkle_root_lane.v" ]; then
  if grep -q 'module k_axi_merkle_root_lane_core' "hardware/fpga/k_axi_merkle_root_lane.v" \
    && grep -q 'merkle_root_l64' "hardware/fpga/k_axi_merkle_root_lane.v" \
    && grep -q 'merkle_root_valid' "hardware/fpga/k_axi_merkle_root_lane.v"; then
    MERKLE_ROOT_CORE_RTL_PRESENT=true
  fi
fi

QUANTUM_CONTROLLER_LANE_PRESENT=false
EPI_POWER_ACCUM_RTL_PATH="hardware/fpga/epistemic_power_accumulator.v"
if [ -f "$EPI_POWER_ACCUM_RTL_PATH" ] && grep -q 'quantum_controller_inc' "$EPI_POWER_ACCUM_RTL_PATH"; then
  QUANTUM_CONTROLLER_LANE_PRESENT=true
fi

SIM_STATUS="$SIM_STATUS" \
SIM_RC="$SIM_RC" \
SYNTH_STATUS="$SYNTH_STATUS" \
SYNTH_RC="$SYNTH_RC" \
QCTRL_SIM_STATUS="$QCTRL_SIM_STATUS" \
QCTRL_SIM_RC="$QCTRL_SIM_RC" \
QCTRL_SYNTH_STATUS="$QCTRL_SYNTH_STATUS" \
QCTRL_SYNTH_RC="$QCTRL_SYNTH_RC" \
QLINK_SIM_STATUS="$QLINK_SIM_STATUS" \
QLINK_SIM_RC="$QLINK_SIM_RC" \
KAXI_SIM_STATUS="$KAXI_SIM_STATUS" \
KAXI_SIM_RC="$KAXI_SIM_RC" \
KAXI_SYNTH_STATUS="$KAXI_SYNTH_STATUS" \
KAXI_SYNTH_RC="$KAXI_SYNTH_RC" \
KAXI_RETURN_SIM_STATUS="$KAXI_RETURN_SIM_STATUS" \
KAXI_RETURN_SIM_RC="$KAXI_RETURN_SIM_RC" \
KAXI_RETURN_SYNTH_STATUS="$KAXI_RETURN_SYNTH_STATUS" \
KAXI_RETURN_SYNTH_RC="$KAXI_RETURN_SYNTH_RC" \
EPI_PWR_SIM_STATUS="$EPI_PWR_SIM_STATUS" \
EPI_PWR_SIM_RC="$EPI_PWR_SIM_RC" \
EPI_PWR_SYNTH_STATUS="$EPI_PWR_SYNTH_STATUS" \
EPI_PWR_SYNTH_RC="$EPI_PWR_SYNTH_RC" \
MERKLE_SYNTH_STATUS="$MERKLE_SYNTH_STATUS" \
MERKLE_SYNTH_RC="$MERKLE_SYNTH_RC" \
MERKLE_ROOT_SYNTH_STATUS="$MERKLE_ROOT_SYNTH_STATUS" \
MERKLE_ROOT_SYNTH_RC="$MERKLE_ROOT_SYNTH_RC" \
MERKLE_LANE_PRESENT="$MERKLE_LANE_PRESENT" \
MERKLE_LANE_CORE_RTL_PRESENT="$MERKLE_LANE_CORE_RTL_PRESENT" \
MERKLE_ROOT_CORE_RTL_PRESENT="$MERKLE_ROOT_CORE_RTL_PRESENT" \
QUANTUM_CONTROLLER_LANE_PRESENT="$QUANTUM_CONTROLLER_LANE_PRESENT" \
python3 - <<'PY'
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

def parse_counter_kv(path: Path) -> dict:
    if not path.exists():
        return {}
    kv = {}
    for line in path.read_text(errors="replace").splitlines():
        if "OMEGA_COUNTERS" not in line:
            continue
        for match in re.finditer(r"([A-Za-z0-9_]+)=(-?[0-9]+)", line):
            kv[match.group(1)] = int(match.group(2))
    return kv

def parse_yosys_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    modules = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"===\s+([A-Za-z0-9_.$\\]+)\s+===", line)
        if not m:
            i += 1
            continue
        module_name = m.group(1).lstrip("\\")
        stats = {}
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith("==="):
                break
            w = re.match(r"Number of wires:\s+([0-9]+)$", s)
            wb = re.match(r"Number of wire bits:\s+([0-9]+)$", s)
            c = re.match(r"Number of cells:\s+([0-9]+)$", s)
            if w:
                stats["wires"] = int(w.group(1))
            if wb:
                stats["wire_bits"] = int(wb.group(1))
            if c:
                stats["cells"] = int(c.group(1))
            j += 1
        if {"wires", "wire_bits", "cells"}.issubset(stats):
            modules[module_name] = stats
        i = j
    return modules

kaxi_counters = parse_counter_kv(Path("artifacts/fpga/vvp_k_axi.log"))
accum_counters = parse_counter_kv(Path("artifacts/fpga/vvp_epistemic_power.log"))
qlink_counters = parse_counter_kv(Path("artifacts/fpga/vvp_quantum_accumulator_link.log"))
kaxi_return_counters = parse_counter_kv(Path("artifacts/fpga/vvp_k_axi_return.log"))

counter_fingerprint = {
    "kaxi_gum": int(kaxi_counters.get("kaxi_gum", -1)),
    "kaxi_prov": int(kaxi_counters.get("kaxi_prov", -1)),
    "kaxi_formal": int(kaxi_counters.get("kaxi_formal", -1)),
    "accum_log": int(accum_counters.get("accum_log", -1)),
    "accum_tx": int(accum_counters.get("accum_tx", -1)),
    "qlink_seen": int(qlink_counters.get("qlink_seen", -1)),
    "qlink_log": int(qlink_counters.get("qlink_log", -1)),
    "qlink_tx": int(qlink_counters.get("qlink_tx", -1)),
    "kaxi_return_seen": int(kaxi_return_counters.get("kaxi_return_seen", -1)),
    "kaxi_return_dropped": int(kaxi_return_counters.get("kaxi_return_dropped", -1)),
    "kaxi_return_overflow": int(kaxi_return_counters.get("kaxi_return_overflow", -1)),
}
counter_complete = all(v >= 0 for v in counter_fingerprint.values())
counter_repro_path = Path("artifacts/fpga/hardware_counter_repro.v1.json")
counter_history_path = Path("artifacts/fpga/hardware_counter_repro_history.v1.json")
previous_fingerprint = None
if counter_repro_path.exists():
    try:
        prev = json.loads(counter_repro_path.read_text())
        if isinstance(prev, dict):
            prev_fp = prev.get("current_fingerprint")
            if isinstance(prev_fp, dict):
                previous_fingerprint = {
                    key: int(prev_fp.get(key, -1)) for key in counter_fingerprint.keys()
                }
    except Exception:
        previous_fingerprint = None

previous_complete = previous_fingerprint is not None and all(
    value >= 0 for value in previous_fingerprint.values()
)

if not counter_complete:
    counter_repro_status = "incomplete"
    reproducible_against_previous = None
elif previous_fingerprint is None or not previous_complete:
    counter_repro_status = "bootstrap"
    reproducible_against_previous = None
else:
    reproducible_against_previous = previous_fingerprint == counter_fingerprint
    counter_repro_status = "pass" if reproducible_against_previous else "drift"

counter_repro = {
    "schema": "sounio.omega.hardware-counter-repro.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "current_fingerprint": counter_fingerprint,
    "previous_fingerprint": previous_fingerprint,
    "reproducible_against_previous": reproducible_against_previous,
    "status": counter_repro_status,
}
counter_repro_path.write_text(json.dumps(counter_repro, indent=2))

history = {
    "schema": "sounio.omega.hardware-counter-repro-history.v1",
    "runs": [],
}
if counter_history_path.exists():
    try:
        obj = json.loads(counter_history_path.read_text())
        if isinstance(obj, dict) and obj.get("schema") == history["schema"] and isinstance(obj.get("runs"), list):
            history = obj
    except Exception:
        history = {
            "schema": "sounio.omega.hardware-counter-repro-history.v1",
            "runs": [],
        }
history["runs"].append(
    {
        "generated_at_utc": counter_repro["generated_at_utc"],
        "fingerprint": counter_fingerprint,
        "status": counter_repro_status,
    }
)
history["runs"] = history["runs"][-32:]
counter_history_path.write_text(json.dumps(history, indent=2))

resource_snapshot = {}
for log_path in (
    Path("artifacts/fpga/yosys.log"),
    Path("artifacts/fpga/yosys_k_axi.log"),
    Path("artifacts/fpga/yosys_k_axi_return.log"),
    Path("artifacts/fpga/yosys_epistemic_power.log"),
    Path("artifacts/fpga/yosys_quantum_controller.log"),
    Path("artifacts/fpga/yosys_k_axi_merkle_lane.log"),
    Path("artifacts/fpga/yosys_k_axi_merkle_root_lane.log"),
):
    resource_snapshot.update(parse_yosys_stats(log_path))

resource_trend_status = "incomplete"
resource_trend_path = Path("artifacts/fpga/hardware_resource_trend.v1.json")
resource_trend_v2_path = Path("artifacts/fpga/hardware_resource_trend.v2.json")
resource_trend = {
    "schema": "sounio.omega.hardware-resource-trend.v1",
    "runs": [],
}
if resource_trend_path.exists():
    try:
        obj = json.loads(resource_trend_path.read_text())
        if isinstance(obj, dict) and obj.get("schema") == resource_trend["schema"] and isinstance(obj.get("runs"), list):
            resource_trend = obj
    except Exception:
        resource_trend = {
            "schema": "sounio.omega.hardware-resource-trend.v1",
            "runs": [],
        }
resource_trend["runs"].append(
    {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "modules": resource_snapshot,
    }
)
resource_trend["runs"] = resource_trend["runs"][-64:]
resource_trend_path.write_text(json.dumps(resource_trend, indent=2))

resource_trend_v2 = {
    "schema": "sounio.omega.hardware-resource-trend.v2",
    "drift_threshold": 0.05,
    "runs": [],
}
if resource_trend_v2_path.exists():
    try:
        obj = json.loads(resource_trend_v2_path.read_text())
        if (
            isinstance(obj, dict)
            and obj.get("schema") == resource_trend_v2["schema"]
            and isinstance(obj.get("runs"), list)
        ):
            resource_trend_v2 = obj
    except Exception:
        resource_trend_v2 = {
            "schema": "sounio.omega.hardware-resource-trend.v2",
            "drift_threshold": 0.05,
            "runs": [],
        }

drift_threshold = float(resource_trend_v2.get("drift_threshold", 0.05))
previous_modules = None
if resource_trend_v2["runs"]:
    last = resource_trend_v2["runs"][-1]
    if isinstance(last, dict):
        maybe_modules = last.get("modules")
        if isinstance(maybe_modules, dict):
            previous_modules = maybe_modules

max_relative_drift = 0.0
compared_points = 0
new_modules = []
removed_modules = []
if previous_modules is not None:
    prev_keys = set(previous_modules.keys())
    curr_keys = set(resource_snapshot.keys())
    new_modules = sorted(curr_keys - prev_keys)
    removed_modules = sorted(prev_keys - curr_keys)
    shared = sorted(prev_keys & curr_keys)
    for module_name in shared:
        prev_stats = previous_modules.get(module_name)
        curr_stats = resource_snapshot.get(module_name)
        if not isinstance(prev_stats, dict) or not isinstance(curr_stats, dict):
            continue
        for field in ("wires", "wire_bits", "cells"):
            prev_value = prev_stats.get(field)
            curr_value = curr_stats.get(field)
            if not isinstance(prev_value, int) or not isinstance(curr_value, int):
                continue
            denom = max(abs(prev_value), 1)
            drift = abs(curr_value - prev_value) / float(denom)
            if drift > max_relative_drift:
                max_relative_drift = drift
            compared_points += 1

if not resource_snapshot:
    resource_trend_status = "incomplete"
elif previous_modules is None:
    resource_trend_status = "bootstrap"
elif max_relative_drift > drift_threshold:
    resource_trend_status = "drift"
else:
    resource_trend_status = "pass"

resource_trend_v2["runs"].append(
    {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "modules": resource_snapshot,
        "status": resource_trend_status,
        "drift_threshold": drift_threshold,
        "max_relative_drift": max_relative_drift,
        "compared_points": compared_points,
        "new_modules": new_modules,
        "removed_modules": removed_modules,
    }
)
resource_trend_v2["runs"] = resource_trend_v2["runs"][-64:]
resource_trend_v2["last_status"] = resource_trend_status
resource_trend_v2["last_max_relative_drift"] = max_relative_drift
resource_trend_v2["last_compared_points"] = compared_points
resource_trend_v2_path.write_text(json.dumps(resource_trend_v2, indent=2))

waveform_candidates = [
    "artifacts/fpga/waveforms/tb_epistemic_mac.vcd",
    "artifacts/fpga/waveforms/tb_epistemic_quantum_controller.vcd",
    "artifacts/fpga/waveforms/tb_quantum_accumulator_link.vcd",
    "artifacts/fpga/waveforms/tb_k_axi_bridge.vcd",
    "artifacts/fpga/waveforms/tb_k_axi_bidirectional.vcd",
    "artifacts/fpga/waveforms/tb_epistemic_power_accumulator.vcd",
]
waveforms_present = [path for path in waveform_candidates if Path(path).exists()]

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
    "quantum_accumulator_link_sim_status": os.environ["QLINK_SIM_STATUS"],
    "quantum_accumulator_link_sim_rc": int(os.environ["QLINK_SIM_RC"]),
    "k_axi_sim_status": os.environ["KAXI_SIM_STATUS"],
    "k_axi_sim_rc": int(os.environ["KAXI_SIM_RC"]),
    "k_axi_synth_status": os.environ["KAXI_SYNTH_STATUS"],
    "k_axi_synth_rc": int(os.environ["KAXI_SYNTH_RC"]),
    "k_axi_return_sim_status": os.environ["KAXI_RETURN_SIM_STATUS"],
    "k_axi_return_sim_rc": int(os.environ["KAXI_RETURN_SIM_RC"]),
    "k_axi_return_synth_status": os.environ["KAXI_RETURN_SYNTH_STATUS"],
    "k_axi_return_synth_rc": int(os.environ["KAXI_RETURN_SYNTH_RC"]),
    "epistemic_power_accumulator_sim_status": os.environ["EPI_PWR_SIM_STATUS"],
    "epistemic_power_accumulator_sim_rc": int(os.environ["EPI_PWR_SIM_RC"]),
    "epistemic_power_accumulator_synth_status": os.environ["EPI_PWR_SYNTH_STATUS"],
    "epistemic_power_accumulator_synth_rc": int(os.environ["EPI_PWR_SYNTH_RC"]),
    "merkle_lane_present": os.environ["MERKLE_LANE_PRESENT"] == "true",
    "merkle_lane_core_rtl_present": os.environ["MERKLE_LANE_CORE_RTL_PRESENT"] == "true",
    "merkle_lane_synth_status": os.environ["MERKLE_SYNTH_STATUS"],
    "merkle_lane_synth_rc": int(os.environ["MERKLE_SYNTH_RC"]),
    "merkle_root_core_rtl_present": os.environ["MERKLE_ROOT_CORE_RTL_PRESENT"] == "true",
    "merkle_root_synth_status": os.environ["MERKLE_ROOT_SYNTH_STATUS"],
    "merkle_root_synth_rc": int(os.environ["MERKLE_ROOT_SYNTH_RC"]),
    "quantum_controller_lane_present": os.environ["QUANTUM_CONTROLLER_LANE_PRESENT"] == "true",
    "hardware_counter_repro_status": counter_repro_status,
    "hardware_counter_fingerprint": counter_fingerprint,
    "hardware_resource_trend_status": resource_trend_status,
    "hardware_resource_trend_threshold": drift_threshold,
    "hardware_resource_trend_max_relative_drift": max_relative_drift,
    "hardware_resource_trend_compared_points": compared_points,
    "hardware_resource_modules": sorted(resource_snapshot.keys()),
    "waveform_paths": waveforms_present,
    "waveforms_present": len(waveforms_present) > 0,
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

if [ "$REQUIRE_QUANTUM_ACCUM_LINK" = "1" ]; then
  if [ "$QLINK_SIM_STATUS" != "pass" ]; then
    echo "error: quantum->accumulator link strict gate failed (sim=$QLINK_SIM_STATUS)" >&2
    exit 2
  fi
fi

if [ "$REQUIRE_K_AXI" = "1" ]; then
  if [ "$KAXI_SIM_STATUS" != "pass" ] || [ "$KAXI_SYNTH_STATUS" != "pass" ]; then
    echo "error: K-AXI strict gate failed (sim=$KAXI_SIM_STATUS synth=$KAXI_SYNTH_STATUS)" >&2
    exit 2
  fi
fi

if [ "$REQUIRE_MERKLE_ROOT" = "1" ]; then
  if [ "$MERKLE_ROOT_SYNTH_STATUS" != "pass" ]; then
    echo "error: K-AXI Merkle root strict gate failed (synth=$MERKLE_ROOT_SYNTH_STATUS)" >&2
    exit 2
  fi
fi

if [ "$REQUIRE_K_AXI_RETURN" = "1" ]; then
  if [ "$KAXI_RETURN_SIM_STATUS" != "pass" ] || [ "$KAXI_RETURN_SYNTH_STATUS" != "pass" ]; then
    echo "error: K-AXI return strict gate failed (sim=$KAXI_RETURN_SIM_STATUS synth=$KAXI_RETURN_SYNTH_STATUS)" >&2
    exit 2
  fi
fi

if [ "$REQUIRE_EPI_POWER_ACCUM" = "1" ]; then
  if [ "$EPI_PWR_SIM_STATUS" != "pass" ] || [ "$EPI_PWR_SYNTH_STATUS" != "pass" ]; then
    echo "error: epistemic power accumulator strict gate failed (sim=$EPI_PWR_SIM_STATUS synth=$EPI_PWR_SYNTH_STATUS)" >&2
    exit 2
  fi
fi

if [ "$REQUIRE_COUNTER_REPRO" = "1" ]; then
  python3 - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/fpga/hardware_counter_repro.v1.json")
if not path.exists():
    raise SystemExit("missing artifacts/fpga/hardware_counter_repro.v1.json")
payload = json.loads(path.read_text())
status = payload.get("status")
if status not in ("pass", "bootstrap"):
    raise SystemExit(f"hardware counter reproducibility check failed (status={status})")
print(f"hardware counter reproducibility status: {status}")
PY
fi

if [ "$REQUIRE_RESOURCE_TREND" = "1" ]; then
  python3 - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/fpga/hardware_resource_trend.v2.json")
if not path.exists():
    raise SystemExit("missing artifacts/fpga/hardware_resource_trend.v2.json")
payload = json.loads(path.read_text())
runs = payload.get("runs")
if not isinstance(runs, list) or not runs:
    raise SystemExit("hardware resource trend is empty")
last = runs[-1]
modules = last.get("modules") if isinstance(last, dict) else None
if not isinstance(modules, dict) or len(modules) < 3:
    raise SystemExit(f"hardware resource trend missing module stats in latest run: {modules}")
status = payload.get("last_status")
if status not in ("pass", "bootstrap"):
    raise SystemExit(f"hardware resource trend status must be pass/bootstrap (got {status})")
max_drift = payload.get("last_max_relative_drift")
threshold = payload.get("drift_threshold", 0.05)
if not isinstance(max_drift, (int, float)):
    raise SystemExit(f"hardware resource trend max drift missing: {max_drift!r}")
if not isinstance(threshold, (int, float)):
    raise SystemExit(f"hardware resource trend threshold missing: {threshold!r}")
if float(max_drift) > float(threshold):
    raise SystemExit(
        f"hardware resource trend drift violation: max_relative_drift={float(max_drift):.6f} "
        f"> threshold={float(threshold):.6f}"
    )
print(
    "hardware resource trend modules: "
    f"{sorted(modules.keys())} "
    f"status={status} max_relative_drift={float(max_drift):.6f} "
    f"threshold={float(threshold):.6f}"
)
PY
fi

echo "FPGA_EPISTEMIC_SEED_DONE"
