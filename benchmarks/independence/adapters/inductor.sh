#!/usr/bin/env bash
set -euo pipefail

baseline_name="${OMEGA_BASELINE_NAME:-pytorch-inductor}"
report_path="${OMEGA_BASELINE_REPORT:-artifacts/omega/external_baselines/${baseline_name}.v1.json}"
families="${OMEGA_BASELINE_FAMILIES:-dense_linear,attention,epistemic_elementwise,monte_carlo,quantum_parallel}"
samples="${OMEGA_BASELINE_SAMPLES:-50}"
baseline_python="${OMEGA_BASELINE_PYTHON:-python3}"
custom_cmd="${OMEGA_PYTORCH_INDUCTOR_CMD:-}"

if [ -n "$custom_cmd" ]; then
  if bash -lc "$custom_cmd"; then
    if [ -f "$report_path" ]; then
      echo "[inductor-adapter] custom command report accepted at $report_path"
      exit 0
    fi
    echo "[inductor-adapter] custom command succeeded but no report at $report_path; falling back to local probe"
  else
    echo "[inductor-adapter] custom command failed; falling back to local probe"
  fi
fi

"$baseline_python" scripts/omega/omega_external_baseline_probe.py \
  --baseline-name "$baseline_name" \
  --report "$report_path" \
  --families "$families" \
  --samples "$samples"

echo "[inductor-adapter] probe finished at $report_path"
