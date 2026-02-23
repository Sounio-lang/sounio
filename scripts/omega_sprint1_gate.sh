#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> omega sprint1 gate: contracts"
CONTRACT_PATH="benchmarks/independence/contract.v2.json" bash scripts/independence_benchmark_gate.sh

python3 - <<'PY'
import json
from pathlib import Path

manifest = Path("benchmarks/independence/omega_sprint1_baselines.v1.json")
obj = json.loads(manifest.read_text())
if obj.get("schema") != "sounio.independence.baseline-set.v1":
    raise SystemExit("invalid baseline manifest schema")
required = {"cuda-cutlass", "triton", "pytorch-inductor"}
actual = set(obj.get("required_baselines", []))
missing = sorted(required - actual)
if missing:
    raise SystemExit(f"missing required baselines: {missing}")
print("omega baseline manifest ok")
PY

echo "==> omega sprint1 gate: seed L2 cubins"
if [ "${OMEGA_SEED_CUBINS:-1}" = "1" ]; then
  bash scripts/omega/omega_seed_cubin_set.sh artifacts/sass/cubin
else
  echo "omega sprint1 gate: cubin seeding disabled (OMEGA_SEED_CUBINS=0)"
fi

echo "==> omega sprint1 gate: L2 SASS pipeline (fallback-safe)"
SASS_ARGS=(
  --kernel-set scripts/omega/omega_sass_kernel_set.json
  --out-dir artifacts/sass/omega
  --report artifacts/sass/omega/sass_patch_report.json
)
if [ -n "${OMEGA_SASS_EQ_CMD:-}" ]; then
  SASS_ARGS+=(--equivalence-cmd "${OMEGA_SASS_EQ_CMD}")
fi
if [ -n "${OMEGA_SASS_EQ_SCRIPT:-}" ]; then
  SASS_ARGS+=(--equivalence-script "${OMEGA_SASS_EQ_SCRIPT}")
fi
if [ -z "${OMEGA_SASS_EQ_CMD:-}" ] && [ -z "${OMEGA_SASS_EQ_SCRIPT:-}" ]; then
  DEFAULT_EQ_CMD="python3 scripts/omega/omega_sass_numeric_equivalence.py --kernel {kernel} --baseline {baseline} --candidate {candidate} --samples ${OMEGA_EQ_SAMPLES:-50} --report artifacts/sass/omega/{kernel}.numeric_equivalence.json"
  if [ "${OMEGA_EQ_PROXY_ALLOWED:-1}" = "1" ]; then
    DEFAULT_EQ_CMD="${DEFAULT_EQ_CMD} --proxy-allowed"
  fi
  SASS_ARGS+=(--equivalence-cmd "${DEFAULT_EQ_CMD}")
fi
if [ "${OMEGA_REQUIRE_ALL_PATCHED:-0}" = "1" ]; then
  SASS_ARGS+=(--require-all-patched)
fi
if [ "${OMEGA_REQUIRE_EQ_PASS:-0}" = "1" ]; then
  SASS_ARGS+=(--require-equivalence-pass)
fi
python3 scripts/omega/omega_sass_patch_pipeline.py "${SASS_ARGS[@]}"

echo "==> omega sprint1 gate: L3 FPGA seed"
OMEGA_REQUIRE_FPGA_PASS="${OMEGA_REQUIRE_FPGA_PASS:-0}" \
OMEGA_REQUIRE_QUANTUM_CONTROLLER="${OMEGA_REQUIRE_QUANTUM_CONTROLLER:-0}" \
  bash scripts/run_fpga_epistemic_seed.sh

echo "==> omega sprint1 gate: QIR shim + quantum telemetry"
QT_ARGS=(
  --contract benchmarks/independence/contract.v2.json
  --out-dir artifacts/quantum/omega
)
if [ "${OMEGA_REQUIRE_QUANTUM_CONFORMANCE:-0}" = "1" ]; then
  QT_ARGS+=(--strict)
fi
python3 scripts/omega/omega_quantum_telemetry.py "${QT_ARGS[@]}"

echo "OMEGA_SPRINT1_GATE_PASS"
