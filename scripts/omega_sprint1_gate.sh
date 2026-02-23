#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p artifacts
GATE_LOG="${OMEGA_GATE_LOG:-artifacts/omega_sprint1_gate.log}"
: >"$GATE_LOG"
NO_RUST_MARKER="${OMEGA_NO_RUST_MARKER:-artifacts/.omega_no_rust_start}"
touch "$NO_RUST_MARKER"

echo "==> omega sprint1 gate: contracts"
# Keep independence gate adapter execution aligned with Omega defaults.
RUN_EXTERNAL_BASELINES="${RUN_EXTERNAL_BASELINES:-${OMEGA_RUN_EXTERNAL_BASELINES:-1}}" \
CONTRACT_PATH="benchmarks/independence/contract.v2.json" \
  bash scripts/independence_benchmark_gate.sh

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

if [ "${OMEGA_REQUIRE_QR_ALIAS_REGRESSION:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: QR alias regression"
  QR_ALIAS_LOG="artifacts/omega_qr_alias_regression.log"
  PATH="$ROOT_DIR:$PATH" souc run tests/regression/linalg_qr_alias_test.sio | tee "$QR_ALIAS_LOG"
  if ! rg -q "QR 4x4 \(1000 iters\): PASS" "$QR_ALIAS_LOG"; then
    echo "error: QR alias regression output missing PASS marker" >&2
    exit 2
  fi
  echo "QR_ALIAS_REGRESSION_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_PURE_SOUNIO_KAXI:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: pure Sounio K-AXI emitter"
  python3 - <<'PY'
import re
import sys
from pathlib import Path

kaxi_path = Path("stdlib/hardware/kaxi.sio")
if not kaxi_path.exists():
    print(f"error: canonical K-AXI schema file missing: {kaxi_path}", file=sys.stderr)
    raise SystemExit(2)

text = kaxi_path.read_text()
missing = []

required_patterns = (
    ("hardware KAXI declaration", r"\bhardware\s+KAXI\s*\{"),
    ("hardware EpistemicPowerAccumulator declaration", r"\bhardware\s+EpistemicPowerAccumulator\s*\{"),
    ("native_keyword hardware_publish", r"\bnative_keyword\s*:\s*hardware_publish\b"),
    ("publish intrinsic __knowledge_kaxi_publish", r"\bpublish\s*:\s*__knowledge_kaxi_publish\b"),
)
for label, pattern in required_patterns:
    if not re.search(pattern, text):
        missing.append(label)

for operation in ("fma", "add", "mul", "div", "prop_var"):
    if not re.search(rf"\b{operation}\s*\{{", text):
        missing.append(f"operation {operation}")

for template in (
    "hardware/fpga/k_axi_slave_epistemic.v",
    "hardware/fpga/epistemic_power_accumulator.v",
    "hardware/fpga/epistemic_rt_kaxi_generated.cu",
    "self-hosted/compiler/codegen/hardware/kaxi_adapters.sio",
    "self-hosted/compiler/codegen/hardware/kaxi_publish_adapters.sio",
):
    marker = f"@@TEMPLATE:{template}"
    if marker not in text:
        missing.append(f"template block {marker}")

if missing:
    print(
        "error: canonical K-AXI template-schema validation failed for stdlib/hardware/kaxi.sio; missing required elements:",
        file=sys.stderr,
    )
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(2)

print("canonical K-AXI template-schema validation ok")
PY

  SOUNIO_PURE_KAXI_STRICT_SELFHOST=1 \
  SOUNIO_PURE_EMITTER_TIMEOUT_SECS="${SOUNIO_PURE_EMITTER_TIMEOUT_SECS:-30}" \
  PATH="$ROOT_DIR:$PATH" souc build --target hardware-kaxi --emit verilog,cuda-stub

  for required in \
    stdlib/hardware/kaxi.sio \
    hardware/fpga/k_axi_slave_epistemic.v \
    hardware/fpga/epistemic_power_accumulator.v \
    hardware/fpga/epistemic_rt_kaxi_generated.cu \
    self-hosted/compiler/codegen/hardware/kaxi_adapters.sio \
    self-hosted/compiler/codegen/hardware/kaxi_publish_adapters.sio \
    artifacts/.pure_sounio_kaxi_generated; do
    if [ ! -f "$required" ]; then
      echo "error: pure Sounio K-AXI artifact missing: $required" >&2
      exit 2
    fi
  done

  if ! rg -q "__knowledge_kaxi_publish" hardware/fpga/epistemic_rt_kaxi_generated.cu; then
    echo "error: generated CUDA stub missing __knowledge_kaxi_publish" >&2
    exit 2
  fi

  if ! rg -q "GENERATED_FROM_STDLIB: stdlib/hardware/kaxi.sio" hardware/fpga/k_axi_slave_epistemic.v; then
    echo "error: generated K-AXI slave missing source marker" >&2
    exit 2
  fi

  if ! rg -q "mode=selfhost-emitter" artifacts/.pure_sounio_kaxi_generated; then
    echo "error: pure Sounio K-AXI must run in selfhost-emitter mode under gate" >&2
    exit 2
  fi

  echo "PURE_SOUNIO_KAXI_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_KAXI_ADAPTER_SELF_CHECK:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: K-AXI adapter self-check"
  bash scripts/omega/omega_kaxi_adapter_self_check.sh
  echo "KAXI_ADAPTER_SELF_CHECK_PASS" | tee -a "$GATE_LOG"
else
  echo "omega sprint1 gate: K-AXI adapter self-check disabled (OMEGA_REQUIRE_KAXI_ADAPTER_SELF_CHECK=0)"
fi

if [ "${OMEGA_REQUIRE_HARDWARE_PUBLISH_SELF_CHECK:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: hardware_publish self-check"
  bash scripts/omega/omega_hardware_publish_self_check.sh
  echo "HARDWARE_PUBLISH_SELF_CHECK_PASS" | tee -a "$GATE_LOG"
else
  echo "omega sprint1 gate: hardware_publish self-check disabled (OMEGA_REQUIRE_HARDWARE_PUBLISH_SELF_CHECK=0)"
fi

if [ "${OMEGA_REQUIRE_PTX_LAUNCH_SELF_CHECK:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: PTX launch self-check"
  bash scripts/omega/omega_ptx_launch_self_check.sh
  echo "PTX_LAUNCH_SELF_CHECK_PASS" | tee -a "$GATE_LOG"
else
  echo "omega sprint1 gate: PTX launch self-check disabled (OMEGA_REQUIRE_PTX_LAUNCH_SELF_CHECK=0)"
fi

if [ "${OMEGA_REQUIRE_PTX_LAUNCH_TELEMETRY:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: PTX launch telemetry"
  PTX_TELEMETRY_ARGS=(
    --out-dir artifacts/ptx/omega
  )
  if [ "${OMEGA_REQUIRE_PTX_LAUNCH_TELEMETRY_STRICT:-1}" = "1" ]; then
    PTX_TELEMETRY_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_ptx_launch_telemetry.py "${PTX_TELEMETRY_ARGS[@]}"
  echo "PTX_LAUNCH_TELEMETRY_PASS" | tee -a "$GATE_LOG"
else
  echo "omega sprint1 gate: PTX launch telemetry disabled (OMEGA_REQUIRE_PTX_LAUNCH_TELEMETRY=0)"
fi

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
OMEGA_REQUIRE_QUANTUM_ACCUM_LINK="${OMEGA_REQUIRE_QUANTUM_ACCUM_LINK:-1}" \
OMEGA_REQUIRE_K_AXI="${OMEGA_REQUIRE_K_AXI:-0}" \
OMEGA_REQUIRE_K_AXI_RETURN="${OMEGA_REQUIRE_K_AXI_RETURN:-0}" \
OMEGA_REQUIRE_EPI_POWER_ACCUM="${OMEGA_REQUIRE_EPI_POWER_ACCUM:-1}" \
OMEGA_REQUIRE_COUNTER_REPRO="${OMEGA_REQUIRE_COUNTER_REPRO:-1}" \
OMEGA_REQUIRE_RESOURCE_TREND="${OMEGA_REQUIRE_RESOURCE_TREND:-1}" \
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

if [ "${OMEGA_REQUIRE_HW_EPI_POWER:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: hardware epistemic power live read"
  HW_EPI_CUBIN="${OMEGA_HW_EPI_POWER_CUBIN:-artifacts/sass/omega/epistemic_rt.cubin}"
  if [ ! -f "$HW_EPI_CUBIN" ]; then
    if command -v nvcc >/dev/null 2>&1; then
      mkdir -p "$(dirname "$HW_EPI_CUBIN")"
      nvcc -cubin -arch="${OMEGA_CUDA_ARCH:-sm_80}" \
        crates/souc/src/codegen/gpu/runtime/epistemic_rt.cu \
        -o "$HW_EPI_CUBIN"
    else
      echo "error: nvcc not found and missing $HW_EPI_CUBIN" >&2
      exit 2
    fi
  fi

  HW_EPI_ARGS=(
    --fpga-report artifacts/fpga/fpga_seed_report.json
    --launch-report artifacts/ptx/omega/ptx_launch_report.json
    --cubin "$HW_EPI_CUBIN"
    --out artifacts/fpga/hardware_epistemic_power_live.v1.json
  )
  if [ "${OMEGA_REQUIRE_HW_EPI_POWER_STRICT:-1}" = "1" ]; then
    HW_EPI_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_hw_epi_power_live_read.py "${HW_EPI_ARGS[@]}"
  echo "HW_EPI_POWER_LIVE_READ_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_HW_EPI_POWER_TREND:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: hardware epistemic power trend"
  HW_EPI_TREND_ARGS=(
    --live-report artifacts/fpga/hardware_epistemic_power_live.v1.json
    --trend artifacts/fpga/hardware_epistemic_power_live_trend.v1.json
    --drift-threshold "${OMEGA_HW_EPI_POWER_TREND_DRIFT_THRESHOLD:-0.20}"
    --max-runs "${OMEGA_HW_EPI_POWER_TREND_MAX_RUNS:-120}"
  )
  if [ "${OMEGA_REQUIRE_HW_EPI_POWER_TREND_STRICT:-1}" = "1" ]; then
    HW_EPI_TREND_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_hw_epi_power_trend.py "${HW_EPI_TREND_ARGS[@]}"
  echo "HW_EPI_POWER_TREND_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_SHADOW_AUDIT:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: shadow audit report"
  SHADOW_AUDIT_ARGS=(
    --gate-log "$GATE_LOG"
    --out artifacts/omega/shadow_audit.v1.json
  )
  if [ "${OMEGA_REQUIRE_SHADOW_AUDIT_STRICT:-1}" = "1" ]; then
    SHADOW_AUDIT_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_shadow_audit_report.py "${SHADOW_AUDIT_ARGS[@]}"
  echo "SHADOW_AUDIT_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_RL_READINESS_BRIDGE:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: RL readiness evidence bridge"
  if [ -f "bootstrap/policies/policy.v2.json" ]; then
    RL_POLICY_PATH="bootstrap/policies/policy.v2.json"
  else
    RL_POLICY_PATH="bootstrap/policies/policy.v1.json"
  fi

  RL_BRIDGE_ARGS=(
    --policy "$RL_POLICY_PATH"
    --ptx-report artifacts/ptx/omega/ptx_launch_report.json
    --sass-report artifacts/sass/omega/sass_patch_report.json
    --hw-live-report artifacts/fpga/hardware_epistemic_power_live.v1.json
    --quantum-report artifacts/quantum/omega/quantum_conformance.json
    --shadow-audit-report "${OMEGA_SHADOW_AUDIT_REPORT:-artifacts/omega/shadow_audit.v1.json}"
    --evidence-out "${OMEGA_RL_READINESS_EVIDENCE_OUT:-bootstrap/policies/rl_readiness.evidence.json}"
    --bridge-out "${OMEGA_RL_READINESS_BRIDGE_OUT:-artifacts/omega/rl_readiness_bridge.v1.json}"
    --compile-overhead "${OMEGA_RL_READINESS_COMPILE_OVERHEAD:-0.18}"
  )
  if [ "${OMEGA_REQUIRE_RL_READINESS_BRIDGE_STRICT:-1}" = "1" ]; then
    RL_BRIDGE_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_rl_readiness_bridge.py "${RL_BRIDGE_ARGS[@]}"

  if [ "${OMEGA_REQUIRE_RL_READINESS_STATUS_SMOKE:-1}" = "1" ]; then
    RL_STATUS_LOG="${OMEGA_RL_READINESS_STATUS_LOG:-artifacts/omega_rl_readiness_status.log}"
    RL_POLICY_SMOKE_OUTPUT="${OMEGA_POLICY_SMOKE_OUTPUT:-artifacts/omega/policy_status_smoke.v2.json}"
    RL_POLICY_SMOKE_ENV_PATH="${OMEGA_POLICY_SMOKE_ENV_PATH:-artifacts/omega/policy_smoke.env}"
    OMEGA_POLICY_PREP_SCRIPT="${OMEGA_POLICY_PREP_SCRIPT:-scripts/omega/omega_prepare_policy_smoke.sh}"
    OMEGA_POLICY_SOUC_BIN="${OMEGA_POLICY_SOUC_BIN:-$ROOT_DIR/souc}"
    if [ -x "$OMEGA_POLICY_PREP_SCRIPT" ]; then
      "$OMEGA_POLICY_PREP_SCRIPT" \
        --policy "$RL_POLICY_PATH" \
        --souc "$OMEGA_POLICY_SOUC_BIN" \
        --corpus "${OMEGA_POLICY_TRAIN_CORPUS:-benchmarks/independence}" \
        --out "$RL_POLICY_SMOKE_OUTPUT" \
        --env-out "$RL_POLICY_SMOKE_ENV_PATH"
      if [ -f "$RL_POLICY_SMOKE_ENV_PATH" ]; then
        # shellcheck disable=SC1090
        source "$RL_POLICY_SMOKE_ENV_PATH"
      fi
    fi
    RL_POLICY_STATUS_PATH="${SOUNIO_POLICY_STATUS_PATH:-$RL_POLICY_PATH}"
    PATH="$ROOT_DIR:$PATH" \
    SOUNIO_RL_READINESS_EVIDENCE_PATH="${OMEGA_RL_READINESS_EVIDENCE_OUT:-bootstrap/policies/rl_readiness.evidence.json}" \
    SOUNIO_POLICY_VERIFY_KEY_PATH="${SOUNIO_POLICY_VERIFY_KEY_PATH:-}" \
      souc opt policy status --policy "$RL_POLICY_STATUS_PATH" | tee "$RL_STATUS_LOG"
    if ! rg -q "opt policy readiness: pass" "$RL_STATUS_LOG"; then
      echo "error: RL readiness status did not report pass" >&2
      exit 2
    fi
  fi

  echo "RL_READINESS_BRIDGE_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_RL_READINESS_TREND:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: RL readiness trend"
  RL_TREND_ARGS=(
    --bridge "${OMEGA_RL_READINESS_BRIDGE_OUT:-artifacts/omega/rl_readiness_bridge.v1.json}"
    --trend "${OMEGA_RL_READINESS_TREND_OUT:-artifacts/omega/rl_readiness_trend.v1.json}"
    --max-runs "${OMEGA_RL_READINESS_TREND_MAX_RUNS:-120}"
    --max-gain-drop "${OMEGA_RL_READINESS_MAX_GAIN_DROP:-0.10}"
    --max-overhead-jump "${OMEGA_RL_READINESS_MAX_OVERHEAD_JUMP:-0.05}"
  )
  if [ "${OMEGA_REQUIRE_RL_READINESS_TREND_STRICT:-1}" = "1" ]; then
    RL_TREND_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_rl_readiness_trend.py "${RL_TREND_ARGS[@]}"
  echo "RL_READINESS_TREND_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_EXTERNAL_BASELINE_COLLECTION:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: external baseline collection"
  EXTERNAL_CMD_FILE="${OMEGA_EXTERNAL_BASELINE_CMDS_FILE:-scripts/omega/external_baseline_cmds.env}"
  if [ -f "$EXTERNAL_CMD_FILE" ]; then
    # shellcheck disable=SC1090
    source "$EXTERNAL_CMD_FILE"
  fi
  DEFAULT_BASELINE_PYTHON="python3"
  if [ -x "artifacts/omega/.venv_external_baselines/bin/python" ]; then
    DEFAULT_BASELINE_PYTHON="artifacts/omega/.venv_external_baselines/bin/python"
  fi
  BASELINE_COLLECTION_ARGS=(
    --contract benchmarks/independence/contract.v2.json
    --baseline-manifest benchmarks/independence/omega_sprint1_baselines.v1.json
    --report-dir "${OMEGA_EXTERNAL_BASELINE_REPORT_DIR:-artifacts/omega/external_baselines}"
    --out "${OMEGA_EXTERNAL_BASELINE_COLLECTION_OUT:-artifacts/omega/external_baseline_collection.v1.json}"
    --timeout-secs "${OMEGA_EXTERNAL_BASELINE_TIMEOUT_SECS:-120}"
    --baseline-python "${OMEGA_BASELINE_PYTHON:-$DEFAULT_BASELINE_PYTHON}"
  )
  if [ "${OMEGA_RUN_EXTERNAL_BASELINES:-1}" = "1" ]; then
    BASELINE_COLLECTION_ARGS+=(--execute)
  fi
  if [ "${OMEGA_REQUIRE_EXTERNAL_BASELINE_COLLECTION_STRICT:-0}" = "1" ]; then
    BASELINE_COLLECTION_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_collect_external_baselines.py "${BASELINE_COLLECTION_ARGS[@]}"
  echo "EXTERNAL_BASELINE_COLLECTION_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_PERFORMANCE_SUMMARY:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: performance summary (external ingest + substitution fallback)"
  PERF_ARGS=(
    --contract benchmarks/independence/contract.v2.json
    --baseline-manifest benchmarks/independence/omega_sprint1_baselines.v1.json
    --ptx-report artifacts/ptx/omega/ptx_launch_report.json
    --sass-report artifacts/sass/omega/sass_patch_report.json
    --fpga-report artifacts/fpga/fpga_seed_report.json
    --quantum-report artifacts/quantum/omega/quantum_conformance.json
    --samples "${OMEGA_PERFORMANCE_SUMMARY_SAMPLES:-50}"
    --external-baseline-dir "${OMEGA_EXTERNAL_BASELINE_REPORT_DIR:-artifacts/omega/external_baselines}"
    --external-collection "${OMEGA_EXTERNAL_BASELINE_COLLECTION_OUT:-artifacts/omega/external_baseline_collection.v1.json}"
    --out "${OMEGA_PERFORMANCE_REPORT_PATH:-artifacts/omega/performance_summary.v1.json}"
  )
  if [ "${OMEGA_PERFORMANCE_PREFER_EXTERNAL:-1}" = "1" ]; then
    PERF_ARGS+=(--prefer-external)
  fi
  if [ "${OMEGA_REQUIRE_PERFORMANCE_SUMMARY_STRICT:-1}" = "1" ]; then
    PERF_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_performance_summary.py "${PERF_ARGS[@]}"
  echo "PERFORMANCE_SUMMARY_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_BASELINE_FREEZE:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: baseline freeze"
  BASELINE_FREEZE_ARGS=(
    --contract benchmarks/independence/contract.v2.json
    --baseline-manifest benchmarks/independence/omega_sprint1_baselines.v1.json
    --external-collection "${OMEGA_EXTERNAL_BASELINE_COLLECTION_OUT:-artifacts/omega/external_baseline_collection.v1.json}"
    --performance-summary "${OMEGA_PERFORMANCE_REPORT_PATH:-artifacts/omega/performance_summary.v1.json}"
    --out "${OMEGA_BASELINE_FREEZE_OUT:-artifacts/omega/baseline_freeze.v1.json}"
  )
  if [ "${OMEGA_REQUIRE_BASELINE_FREEZE_STRICT:-1}" = "1" ]; then
    BASELINE_FREEZE_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_baseline_freeze.py "${BASELINE_FREEZE_ARGS[@]}"
  echo "BASELINE_FREEZE_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_POLICY_MODE_GUARD:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: policy mode guard"
  if [ -f "bootstrap/policies/policy.v2.json" ]; then
    POLICY_GUARD_PATH="bootstrap/policies/policy.v2.json"
  else
    POLICY_GUARD_PATH="bootstrap/policies/policy.v1.json"
  fi

  POLICY_GUARD_ARGS=(
    --policy "$POLICY_GUARD_PATH"
    --bridge "${OMEGA_RL_READINESS_BRIDGE_OUT:-artifacts/omega/rl_readiness_bridge.v1.json}"
    --rl-trend "${OMEGA_RL_READINESS_TREND_OUT:-artifacts/omega/rl_readiness_trend.v1.json}"
    --min-stable-runs "${OMEGA_POLICY_ACTIVE_MIN_STABLE_RUNS:-5}"
    --out "${OMEGA_POLICY_MODE_GUARD_OUT:-artifacts/omega/policy_mode_guard.v1.json}"
    --strict
  )
  if [ "${OMEGA_ALLOW_POLICY_ACTIVE:-0}" = "1" ]; then
    POLICY_GUARD_ARGS+=(--allow-active)
  fi
  python3 scripts/omega/omega_policy_mode_guard.py "${POLICY_GUARD_ARGS[@]}"
  echo "POLICY_MODE_GUARD_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_RL_READINESS_REPLAY:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: RL readiness deterministic replay"
  RL_REPLAY_ARGS=(
    --iterations "${OMEGA_RL_READINESS_REPLAY_ITERATIONS:-5}"
    --bridge-script scripts/omega/omega_rl_readiness_bridge.py
    --trend-script scripts/omega/omega_rl_readiness_trend.py
    --policy "${RL_POLICY_PATH:-bootstrap/policies/policy.v2.json}"
    --ptx-report artifacts/ptx/omega/ptx_launch_report.json
    --sass-report artifacts/sass/omega/sass_patch_report.json
    --hw-live-report artifacts/fpga/hardware_epistemic_power_live.v1.json
    --quantum-report artifacts/quantum/omega/quantum_conformance.json
    --shadow-audit-report "${OMEGA_SHADOW_AUDIT_REPORT:-artifacts/omega/shadow_audit.v1.json}"
    --compile-overhead "${OMEGA_RL_READINESS_COMPILE_OVERHEAD:-0.18}"
    --bridge "${OMEGA_RL_READINESS_REPLAY_BRIDGE_OUT:-artifacts/omega/rl_readiness_bridge.replay.v1.json}"
    --trend "${OMEGA_RL_READINESS_REPLAY_TREND_OUT:-artifacts/omega/rl_readiness_replay_trend.v1.json}"
    --out "${OMEGA_RL_READINESS_REPLAY_OUT:-artifacts/omega/rl_readiness_replay.v1.json}"
  )
  if [ "${OMEGA_REQUIRE_RL_READINESS_REPLAY_STRICT:-1}" = "1" ]; then
    RL_REPLAY_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_rl_readiness_replay.py "${RL_REPLAY_ARGS[@]}"
  echo "RL_READINESS_REPLAY_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_GOVERNANCE_ATTESTATION:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: governance attestation"
  GOV_ATTEST_ARGS=(
    --out "${OMEGA_GOVERNANCE_ATTESTATION_OUT:-artifacts/omega/governance_attestation.v1.json}"
  )
  REQUIRE_GOV_ATTEST_SIG="${OMEGA_REQUIRE_GOVERNANCE_ATTEST_SIGNATURE:-1}"
  if [ "$REQUIRE_GOV_ATTEST_SIG" = "1" ] && [ -z "${OMEGA_GOVERNANCE_ATTEST_PRIVATE_KEY:-}" ]; then
    AUTO_KEY_PATH="${OMEGA_GOVERNANCE_ATTEST_AUTO_PRIVATE_KEY:-artifacts/omega/governance_attest_ed25519.pem}"
    if [ ! -f "$AUTO_KEY_PATH" ]; then
      if ! command -v openssl >/dev/null 2>&1; then
        echo "error: openssl is required for governance attestation signing" >&2
        exit 2
      fi
      mkdir -p "$(dirname "$AUTO_KEY_PATH")"
      openssl genpkey -algorithm ED25519 -out "$AUTO_KEY_PATH"
      chmod 600 "$AUTO_KEY_PATH"
    fi
    OMEGA_GOVERNANCE_ATTEST_PRIVATE_KEY="$AUTO_KEY_PATH"
    if [ -z "${OMEGA_GOVERNANCE_ATTEST_KEY_ID:-}" ]; then
      OMEGA_GOVERNANCE_ATTEST_KEY_ID="auto-local-ed25519"
    fi
  fi
  if [ -n "${OMEGA_GOVERNANCE_ATTEST_PRIVATE_KEY:-}" ]; then
    GOV_ATTEST_ARGS+=(--private-key "${OMEGA_GOVERNANCE_ATTEST_PRIVATE_KEY}")
  fi
  if [ -n "${OMEGA_GOVERNANCE_ATTEST_KEY_ID:-}" ]; then
    GOV_ATTEST_ARGS+=(--key-id "${OMEGA_GOVERNANCE_ATTEST_KEY_ID}")
  fi
  if [ "$REQUIRE_GOV_ATTEST_SIG" = "1" ]; then
    GOV_ATTEST_ARGS+=(--require-signature)
  fi
  if [ "${OMEGA_REQUIRE_GOVERNANCE_ATTESTATION_STRICT:-1}" = "1" ]; then
    GOV_ATTEST_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_governance_attest.py "${GOV_ATTEST_ARGS[@]}"
  echo "GOVERNANCE_ATTESTATION_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_SPRINT1_RELEASE_READINESS:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: sprint1 release readiness (hard+rollover)"
  if [ -f "bootstrap/policies/policy.v2.json" ]; then
    RELEASE_POLICY_PATH="bootstrap/policies/policy.v2.json"
  else
    RELEASE_POLICY_PATH="bootstrap/policies/policy.v1.json"
  fi
  RELEASE_ARGS=(
    --contract benchmarks/independence/contract.v2.json
    --policy "$RELEASE_POLICY_PATH"
    --ptx-report artifacts/ptx/omega/ptx_launch_report.json
    --sass-report artifacts/sass/omega/sass_patch_report.json
    --fpga-report artifacts/fpga/fpga_seed_report.json
    --quantum-report artifacts/quantum/omega/quantum_conformance.json
    --rl-bridge "${OMEGA_RL_READINESS_BRIDGE_OUT:-artifacts/omega/rl_readiness_bridge.v1.json}"
    --policy-guard "${OMEGA_POLICY_MODE_GUARD_OUT:-artifacts/omega/policy_mode_guard.v1.json}"
    --performance-report "${OMEGA_PERFORMANCE_REPORT_PATH:-artifacts/omega/performance_summary.v1.json}"
    --out "${OMEGA_SPRINT1_RELEASE_READINESS_OUT:-artifacts/omega/sprint1_release_readiness.v1.json}"
  )
  if [ "${OMEGA_REQUIRE_SPRINT1_RELEASE_READINESS_STRICT:-1}" = "1" ]; then
    RELEASE_ARGS+=(--strict)
  fi
  if [ "${OMEGA_REQUIRE_SPRINT1_RELEASE_READY_NOW:-0}" = "1" ]; then
    RELEASE_ARGS+=(--require-release-ready)
  fi
  if [ "${OMEGA_REQUIRE_SPRINT1_SUCCESS_CRITERIA_NOW:-1}" = "1" ]; then
    RELEASE_ARGS+=(--require-success-criteria)
  fi
  python3 scripts/omega/omega_release_readiness.py "${RELEASE_ARGS[@]}"
  echo "SPRINT1_RELEASE_READINESS_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_HW_TELEMETRY_REGRESSION:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: hardware telemetry regression"
  OMEGA_REQUIRE_HW_EPI_POWER="${OMEGA_REQUIRE_HW_EPI_POWER:-1}"
  OMEGA_REQUIRE_HW_EPI_POWER_TREND="${OMEGA_REQUIRE_HW_EPI_POWER_TREND:-1}"
  OMEGA_REQUIRE_RL_READINESS_BRIDGE="${OMEGA_REQUIRE_RL_READINESS_BRIDGE:-1}"
  OMEGA_REQUIRE_SHADOW_AUDIT="${OMEGA_REQUIRE_SHADOW_AUDIT:-1}"
  OMEGA_REQUIRE_RL_READINESS_TREND="${OMEGA_REQUIRE_RL_READINESS_TREND:-1}"
  OMEGA_REQUIRE_EXTERNAL_BASELINE_COLLECTION="${OMEGA_REQUIRE_EXTERNAL_BASELINE_COLLECTION:-1}"
  OMEGA_REQUIRE_PERFORMANCE_SUMMARY="${OMEGA_REQUIRE_PERFORMANCE_SUMMARY:-1}"
  OMEGA_REQUIRE_BASELINE_FREEZE="${OMEGA_REQUIRE_BASELINE_FREEZE:-1}"
  OMEGA_REQUIRE_POLICY_MODE_GUARD="${OMEGA_REQUIRE_POLICY_MODE_GUARD:-1}"
  OMEGA_REQUIRE_RL_READINESS_REPLAY="${OMEGA_REQUIRE_RL_READINESS_REPLAY:-1}"
  OMEGA_REQUIRE_GOVERNANCE_ATTESTATION="${OMEGA_REQUIRE_GOVERNANCE_ATTESTATION:-1}"
  OMEGA_REQUIRE_SPRINT1_RELEASE_READINESS="${OMEGA_REQUIRE_SPRINT1_RELEASE_READINESS:-1}"
  OMEGA_REQUIRE_SPRINT1_SUCCESS_CRITERIA_NOW="${OMEGA_REQUIRE_SPRINT1_SUCCESS_CRITERIA_NOW:-1}"
  HW_TELEMETRY_ARGS=()
  if [ "${OMEGA_HW_TELEMETRY_REGRESSION_STRICT:-0}" = "1" ]; then
    HW_TELEMETRY_ARGS+=(--strict)
  fi
  OMEGA_REQUIRE_HW_EPI_POWER="$OMEGA_REQUIRE_HW_EPI_POWER" \
  OMEGA_REQUIRE_HW_EPI_POWER_TREND="$OMEGA_REQUIRE_HW_EPI_POWER_TREND" \
  OMEGA_REQUIRE_RL_READINESS_BRIDGE="$OMEGA_REQUIRE_RL_READINESS_BRIDGE" \
  OMEGA_REQUIRE_SHADOW_AUDIT="$OMEGA_REQUIRE_SHADOW_AUDIT" \
  OMEGA_REQUIRE_RL_READINESS_TREND="$OMEGA_REQUIRE_RL_READINESS_TREND" \
  OMEGA_REQUIRE_EXTERNAL_BASELINE_COLLECTION="$OMEGA_REQUIRE_EXTERNAL_BASELINE_COLLECTION" \
  OMEGA_REQUIRE_PERFORMANCE_SUMMARY="$OMEGA_REQUIRE_PERFORMANCE_SUMMARY" \
  OMEGA_REQUIRE_BASELINE_FREEZE="$OMEGA_REQUIRE_BASELINE_FREEZE" \
  OMEGA_REQUIRE_POLICY_MODE_GUARD="$OMEGA_REQUIRE_POLICY_MODE_GUARD" \
  OMEGA_REQUIRE_RL_READINESS_REPLAY="$OMEGA_REQUIRE_RL_READINESS_REPLAY" \
  OMEGA_REQUIRE_GOVERNANCE_ATTESTATION="$OMEGA_REQUIRE_GOVERNANCE_ATTESTATION" \
  OMEGA_REQUIRE_SPRINT1_RELEASE_READINESS="$OMEGA_REQUIRE_SPRINT1_RELEASE_READINESS" \
  OMEGA_REQUIRE_SPRINT1_SUCCESS_CRITERIA_NOW="$OMEGA_REQUIRE_SPRINT1_SUCCESS_CRITERIA_NOW" \
    python3 scripts/omega/omega_hardware_telemetry_regression.py "${HW_TELEMETRY_ARGS[@]}"
  echo "HW_TELEMETRY_REGRESSION_PASS" | tee -a "$GATE_LOG"
else
  echo "omega sprint1 gate: hardware telemetry regression disabled (OMEGA_REQUIRE_HW_TELEMETRY_REGRESSION=0)"
fi

if [ "${OMEGA_REQUIRE_NO_RUST:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: no-rust mode guard"
  mapfile -t changed_rust < <(find crates -type f -name '*.rs' -newer "$NO_RUST_MARKER" | sort)
  if [ "${#changed_rust[@]}" -ne 0 ]; then
    echo "error: no-rust mode violated; Rust files modified during gate run:" >&2
    printf '  %s\n' "${changed_rust[@]}" >&2
    exit 2
  fi
  echo "No-Rust mode PASS" | tee -a "$GATE_LOG"
fi

echo "OMEGA_SPRINT1_GATE_PASS"
