#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p artifacts
GATE_LOG="${OMEGA_GATE_LOG:-artifacts/omega_sprint1_gate.log}"
: >"$GATE_LOG"
NO_RUST_MARKER="${OMEGA_NO_RUST_MARKER:-artifacts/.omega_no_rust_start}"
touch "$NO_RUST_MARKER"

if [ -x "scripts/omega/omega_resolve_souc_bin.sh" ]; then
  RESOLVED_SOUC_BIN="$(
    OMEGA_SOUC_REQUIRE_PINNED="${OMEGA_SOUC_REQUIRE_PINNED:-1}" \
    OMEGA_SOUC_ALLOW_LOCAL_FALLBACK="${OMEGA_SOUC_ALLOW_LOCAL_FALLBACK:-0}" \
      scripts/omega/omega_resolve_souc_bin.sh --print-path
  )"
else
  RESOLVED_SOUC_BIN="${OMEGA_POLICY_SOUC_BIN:-$ROOT_DIR/souc}"
fi
export OMEGA_POLICY_SOUC_BIN="$RESOLVED_SOUC_BIN"
export SOUC_BIN="$RESOLVED_SOUC_BIN"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

echo "==> omega sprint1 gate: contracts"
# Keep independence gate adapter execution aligned with Omega defaults.
RUN_EXTERNAL_BASELINES="${RUN_EXTERNAL_BASELINES:-${OMEGA_RUN_EXTERNAL_BASELINES:-1}}" \
CONTRACT_PATH="benchmarks/independence/contract.v2.json" \
  bash scripts/independence_benchmark_gate.sh
echo "INPLACE_POLICY_SIGN_PASS" | tee -a "$GATE_LOG"

if [ "${OMEGA_REQUIRE_CANONICAL_KEY_BOOTSTRAP:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: canonical key bootstrap"
  CANONICAL_BOOTSTRAP_SCRIPT="${OMEGA_CANONICAL_BOOTSTRAP_SCRIPT:-scripts/omega_canonical_key_bootstrap.sh}"
  CANONICAL_ENV_PATH="${OMEGA_CANONICAL_ENV_OUT:-artifacts/omega/canonical_key.env}"
  if [ ! -x "$CANONICAL_BOOTSTRAP_SCRIPT" ]; then
    echo "error: canonical key bootstrap script missing or not executable: $CANONICAL_BOOTSTRAP_SCRIPT" >&2
    exit 2
  fi
  "$CANONICAL_BOOTSTRAP_SCRIPT" --env-out "$CANONICAL_ENV_PATH"
  if [ ! -f "$CANONICAL_ENV_PATH" ]; then
    echo "error: canonical key env file missing: $CANONICAL_ENV_PATH" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$CANONICAL_ENV_PATH"
  if [ ! -f "${OMEGA_CANONICAL_PRIVKEY:-}" ] || [ ! -f "${OMEGA_CANONICAL_PUBKEY:-}" ]; then
    echo "error: canonical key bootstrap did not export valid key paths" >&2
    exit 2
  fi
  echo "CANONICAL_KEY_BOOTSTRAP_PASS" | tee -a "$GATE_LOG"

  if [ "${OMEGA_REQUIRE_INPLACE_POLICY_SIGN:-1}" = "1" ]; then
    echo "==> omega sprint1 gate: in-place policy sign"
    SIGN_POLICY_PATH="${OMEGA_SIGN_POLICY_PATH:-bootstrap/policies/policy.v2.json}"
    SIGN_SOUC_BIN="${OMEGA_POLICY_SOUC_BIN:-$ROOT_DIR/souc}"
    if [ -x "$SIGN_SOUC_BIN" ] && "$SIGN_SOUC_BIN" opt policy sign --help >/dev/null 2>&1; then
      SOUNIO_POLICY_SIGNING_KEY_PATH="$OMEGA_CANONICAL_PRIVKEY" \
      SOUNIO_POLICY_VERIFY_KEY_PATH="$OMEGA_CANONICAL_PUBKEY" \
        "$SIGN_SOUC_BIN" opt policy sign --policy "$SIGN_POLICY_PATH"
      SOUNIO_POLICY_VERIFY_KEY_PATH="$OMEGA_CANONICAL_PUBKEY" \
        "$SIGN_SOUC_BIN" opt policy eval --policy "$SIGN_POLICY_PATH"
      echo "INPLACE_POLICY_SIGN_PASS" | tee -a "$GATE_LOG"
    else
      echo "warning: souc opt policy sign not available; skipping in-place sign gate"
    fi
  fi

  if [ "${OMEGA_REQUIRE_PINNED_DIGEST:-1}" = "1" ]; then
    echo "==> omega sprint1 gate: pinned digest policy check"
    PIN_CHECK_POLICY_PATH="${OMEGA_SIGN_POLICY_PATH:-bootstrap/policies/policy.v2.json}"
    if [ ! -f "$PIN_CHECK_POLICY_PATH" ] && [ -f "bootstrap/policies/policy.v1.json" ]; then
      PIN_CHECK_POLICY_PATH="bootstrap/policies/policy.v1.json"
    fi
    PIN_CHECK_SOUC_BIN="${OMEGA_POLICY_SOUC_BIN:-$ROOT_DIR/souc}"
    set +e
    PIN_CHECK_OUTPUT="$(
      SOUNIO_POLICY_VERIFY_KEY_PATH="$OMEGA_CANONICAL_PUBKEY" \
        "$PIN_CHECK_SOUC_BIN" opt policy status --policy "$PIN_CHECK_POLICY_PATH" 2>&1
    )"
    PIN_CHECK_RC=$?
    set -e
    echo "$PIN_CHECK_OUTPUT"
    if [ "$PIN_CHECK_RC" -ne 0 ]; then
      echo "error: pinned digest policy status command failed: policy=$PIN_CHECK_POLICY_PATH" >&2
      exit "$PIN_CHECK_RC"
    fi
    if ! grep -q "signature=verified" <<<"$PIN_CHECK_OUTPUT"; then
      echo "error: pinned digest policy check requires signature=verified" >&2
      exit 2
    fi
    if ! grep -q "opt policy pinned: status=verified" <<<"$PIN_CHECK_OUTPUT"; then
      echo "error: pinned digest policy check requires opt policy pinned: status=verified" >&2
      exit 2
    fi
    echo "PINNED_DIGEST_PASS" | tee -a "$GATE_LOG"
  fi
fi

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
  PATH="$ROOT_DIR:$PATH" "$OMEGA_POLICY_SOUC_BIN" run tests/regression/linalg_qr_alias_test.sio | tee "$QR_ALIAS_LOG"
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

  PURE_KAXI_SOURCE="${OMEGA_PURE_KAXI_SOURCE:-stdlib/hardware/kaxi.sio}"
  SOUNIO_PURE_KAXI_STRICT_SELFHOST=1 \
  SOUNIO_PURE_EMITTER_TIMEOUT_SECS="${SOUNIO_PURE_EMITTER_TIMEOUT_SECS:-30}" \
  PATH="$ROOT_DIR:$PATH" "$OMEGA_POLICY_SOUC_BIN" build --target hardware-kaxi --emit verilog,cuda-stub "$PURE_KAXI_SOURCE"

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
OMEGA_REQUIRE_K_AXI_RETURN="${OMEGA_REQUIRE_K_AXI_RETURN:-1}" \
OMEGA_REQUIRE_EPI_POWER_ACCUM="${OMEGA_REQUIRE_EPI_POWER_ACCUM:-1}" \
OMEGA_REQUIRE_MERKLE_LANE="${OMEGA_REQUIRE_MERKLE_LANE:-1}" \
OMEGA_REQUIRE_MERKLE_ROOT="${OMEGA_REQUIRE_MERKLE_ROOT:-1}" \
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
      HW_EPI_RUNTIME_SRC="${OMEGA_HW_EPI_RUNTIME_SRC:-hardware/fpga/epistemic_rt_kaxi_generated.cu}"
      if [ ! -f "$HW_EPI_RUNTIME_SRC" ]; then
        if [ -f "artifacts/sass/omega/epistemic_rt_generated.cu" ]; then
          HW_EPI_RUNTIME_SRC="artifacts/sass/omega/epistemic_rt_generated.cu"
        elif [ -f "crates/souc/src/codegen/gpu/runtime/epistemic_rt.cu" ]; then
          HW_EPI_RUNTIME_SRC="crates/souc/src/codegen/gpu/runtime/epistemic_rt.cu"
        else
          echo "error: missing hardware runtime source for cubin build (checked hardware/fpga, artifacts/sass, crates)" >&2
          exit 2
        fi
      fi
      nvcc -cubin -arch="${OMEGA_CUDA_ARCH:-sm_80}" \
        "$HW_EPI_RUNTIME_SRC" \
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

if [ "${OMEGA_REQUIRE_DATA_STRUCTURES:-1}" = "1" ]; then
  echo "==> omega sprint2 gate: data structures surface"
  ORDERED_MAP_PATH="${OMEGA_ORDERED_MAP_PATH:-self-hosted/collections/ordered_map.sio}"
  ARENA_PATH="${OMEGA_ARENA_PATH:-self-hosted/collections/arena.sio}"
  INTERNER_PATH="${OMEGA_INTERNER_PATH:-self-hosted/intern.sio}"
  GRAPH_PATH="${OMEGA_GRAPH_PATH:-self-hosted/collections/graph.sio}"
  for required_path in "$ORDERED_MAP_PATH" "$ARENA_PATH" "$INTERNER_PATH" "$GRAPH_PATH"; do
    if [ ! -f "$required_path" ]; then
      echo "error: data structures module missing: $required_path" >&2
      exit 2
    fi
  done
  python3 - "$ORDERED_MAP_PATH" "$ARENA_PATH" "$INTERNER_PATH" "$GRAPH_PATH" <<'PY'
import sys
from pathlib import Path

ordered_map_path, arena_path, interner_path, graph_path = [Path(p) for p in sys.argv[1:5]]

checks = {
    ordered_map_path: (
        "struct OrderedMap",
        "fn ordered_map_new(",
        "fn ordered_map_insert(",
        "fn ordered_map_get(",
        "fn ordered_map_contains(",
        "fn ordered_map_remove(",
        "fn ordered_map_len(",
        "fn ordered_map_key_at(",
        "fn ordered_map_value_at(",
        "keys: [i64; 1024]",
        "values: [i64; 1024]",
    ),
    arena_path: (
        "struct Arena",
        "fn arena_new(",
        "fn arena_alloc(",
        "fn arena_get(",
        "fn arena_set(",
        "fn arena_len(",
        "data: [i64; 8192]",
    ),
    interner_path: (
        "struct StringInterner",
        "fn interner_new(",
        "fn interner_intern(",
        "fn interner_resolve(",
        "fn interner_contains(",
        "buf: [i8; 65536]",
        "offsets: [i64; 4096]",
        "lengths: [i64; 4096]",
        "hashes: [i64; 4096]",
    ),
    graph_path: (
        "struct DiGraph",
        "fn graph_new(",
        "fn graph_add_node(",
        "fn graph_add_edge(",
        "fn graph_has_edge(",
        "fn graph_successors(",
        "fn graph_predecessors(",
        "fn graph_node_count(",
        "fn graph_edge_count(",
        "fn graph_dfs_order(",
        "fn graph_topo_sort(",
        "nodes: [i64; 1024]",
        "edges_src: [i64; 4096]",
        "edges_dst: [i64; 4096]",
    ),
}

total_symbols = 0
for path, symbols in checks.items():
    text = path.read_text()
    for symbol in symbols:
        if symbol not in text:
            raise SystemExit(f"{path}: missing data-structure symbol {symbol!r}")
    total_symbols += len(symbols)

print(
    f"data_structures_check: pass files={len(checks)} "
    f"symbols={total_symbols}"
)
PY
  echo "DATA_STRUCTURES_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_GPU_IR_EXPANSION:-1}" = "1" ]; then
  echo "==> omega sprint2 gate: GPU IR expansion surface"
  GPU_KERNEL_IR_PATH="${OMEGA_GPU_KERNEL_IR_PATH:-self-hosted/gpu/kernel_ir.sio}"
  METAL_CODEGEN_PATH="${OMEGA_METAL_CODEGEN_PATH:-self-hosted/gpu/metal.sio}"
  PTX_ADVANCED_PATH="${OMEGA_PTX_ADVANCED_PATH:-self-hosted/gpu/ptx_advanced.sio}"
  if [ ! -f "$GPU_KERNEL_IR_PATH" ]; then
    echo "error: GPU kernel IR module missing: $GPU_KERNEL_IR_PATH" >&2
    exit 2
  fi
  python3 - "$GPU_KERNEL_IR_PATH" "$METAL_CODEGEN_PATH" "$PTX_ADVANCED_PATH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
metal_path = Path(sys.argv[2])
ptx_path = Path(sys.argv[3])
text = path.read_text()
metal_text = metal_path.read_text()
ptx_text = ptx_path.read_text()
required_symbols = (
    "enum GpuMemorySpace",
    "GpuMemGlobal",
    "GpuMemShared",
    "GpuMemLocal",
    "GpuMemConstant",
    "memory_space: GpuMemorySpace",
    "pred_reg: i64",
    "mask: i64",
    "max_reg_f64: i64",
    "shared_mem_bytes: i64",
    "max_threads: i64",
)
for symbol in required_symbols:
    if symbol not in text:
        raise SystemExit(f"{path}: missing required symbol {symbol!r}")

required_opcodes = (
    "GpuShflUp",
    "GpuShflBfly",
    "GpuShflIdx",
    "GpuVote",
    "GpuBallot",
    "GpuWmma",
    "GpuMma",
    "GpuTmaLoad",
    "GpuTmaStore",
    "GpuLdGlobalCached",
    "GpuStGlobalWriteback",
    "GpuPrefetch",
    "GpuSqrt",
    "GpuRsqrt",
    "GpuExp2",
    "GpuLg2",
    "GpuSin",
    "GpuCos",
    "GpuAbs",
    "GpuRcp",
    "GpuSetpGt",
    "GpuSetpNe",
    "GpuBra",
    "GpuExit",
    "GpuGetLaneId",
    "GpuGetWarpId",
    "GpuGetSmId",
    "GpuGetGridDim",
)
for opcode in required_opcodes:
    if f"{opcode}," not in text:
        raise SystemExit(f"{path}: missing expanded opcode {opcode!r}")
    backend_marker = f"GpuOpcode::{opcode}"
    if backend_marker not in metal_text:
        raise SystemExit(
            f"{metal_path}: missing GPU IR expanded opcode coverage for {opcode!r}"
        )
    if backend_marker not in ptx_text:
        raise SystemExit(
            f"{ptx_path}: missing GPU IR expanded opcode coverage for {opcode!r}"
        )

print(
    "gpu_ir_expansion_check: pass "
    f"path={path} opcodes={len(required_opcodes)} "
    f"metal_coverage={len(required_opcodes)} ptx_coverage={len(required_opcodes)}"
)
PY
  echo "GPU_IR_EXPANSION_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_HLIR_GPU_CROSS_COVERAGE:-1}" = "1" ]; then
  echo "==> omega sprint2 gate: HLIR to GPU cross-coverage"
  HLIR_TO_GPU_PATH="${OMEGA_HLIR_TO_GPU_PATH:-self-hosted/gpu/hlir_to_gpu.sio}"
  METAL_CODEGEN_PATH="${OMEGA_METAL_CODEGEN_PATH:-self-hosted/gpu/metal.sio}"
  PTX_ADVANCED_PATH="${OMEGA_PTX_ADVANCED_PATH:-self-hosted/gpu/ptx_advanced.sio}"
  if [ ! -f "$HLIR_TO_GPU_PATH" ]; then
    echo "error: HLIR->GPU lowering module missing: $HLIR_TO_GPU_PATH" >&2
    exit 2
  fi
  python3 - "$HLIR_TO_GPU_PATH" "$METAL_CODEGEN_PATH" "$PTX_ADVANCED_PATH" <<'PY'
import sys
from pathlib import Path

hlir_path = Path(sys.argv[1])
metal_path = Path(sys.argv[2])
ptx_path = Path(sys.argv[3])
hlir_text = hlir_path.read_text()
metal_text = metal_path.read_text()
ptx_text = ptx_path.read_text()

shared_core_pairs = (
    ("HlirOpAdd", "GpuOpcodeAdd", "GpuOpcode::GpuAdd"),
    ("HlirOpSub", "GpuOpcodeSub", "GpuOpcode::GpuSub"),
    ("HlirOpMul", "GpuOpcodeMul", "GpuOpcode::GpuMul"),
    ("HlirOpDiv", "GpuOpcodeDiv", "GpuOpcode::GpuDiv"),
    ("HlirOpEq", "GpuOpcodeSetpEq", "GpuOpcode::GpuSetpEq"),
    ("HlirOpNe", "GpuOpcodeSetpNe", "GpuOpcode::GpuSetpNe"),
    ("HlirOpLt", "GpuOpcodeSetpLt", "GpuOpcode::GpuSetpLt"),
    ("HlirOpLe", "GpuOpcodeSetpLe", "GpuOpcode::GpuSetpLe"),
    ("HlirOpGt", "GpuOpcodeSetpGt", "GpuOpcode::GpuSetpGt"),
    ("HlirOpGe", "GpuOpcodeSetpGe", "GpuOpcode::GpuSetpGe"),
    ("HlirOpLoad", "GpuOpcodeLoadGlobal", "GpuOpcode::GpuLoadGlobal"),
    ("HlirOpStore", "GpuOpcodeStoreGlobal", "GpuOpcode::GpuStoreGlobal"),
    ("HlirOpSExt", "GpuOpcodeCvt", "GpuOpcode::GpuCvt"),
    ("HlirOpSelect", "GpuOpcodeSelp", "GpuOpcode::GpuSelp"),
    ("HlirOpFMA", "GpuOpcodeFma", "GpuOpcode::GpuFma"),
    ("HlirOpGetThreadId", "GpuOpcodeGetTid", "GpuOpcode::GpuGetTid"),
    ("HlirOpGetBlockId", "GpuOpcodeGetBid", "GpuOpcode::GpuGetBid"),
    ("HlirOpGetBlockDim", "GpuOpcodeGetNtid", "GpuOpcode::GpuGetNtid"),
    ("HlirOpAtomicAdd", "GpuOpcodeAtomicAdd", "GpuOpcode::GpuAtomicAdd"),
    ("HlirOpSyncThreads", "GpuOpcodeBarrierSync", "GpuOpcode::GpuBarrierSync"),
)

for hlir_op, gpu_const, backend_marker in shared_core_pairs:
    if f"op_tag == {hlir_op}" not in hlir_text:
        raise SystemExit(f"{hlir_path}: missing HLIR lowering branch for {hlir_op}")
    if gpu_const not in hlir_text:
        raise SystemExit(
            f"{hlir_path}: missing GPU opcode constant marker {gpu_const!r} for {hlir_op}"
        )
    if backend_marker not in metal_text:
        raise SystemExit(
            f"{metal_path}: missing shared-core backend marker {backend_marker!r}"
        )
    if backend_marker not in ptx_text:
        raise SystemExit(
            f"{ptx_path}: missing shared-core backend marker {backend_marker!r}"
        )

if "fn hlir_lower_instr(" not in hlir_text:
    raise SystemExit(f"{hlir_path}: missing fn hlir_lower_instr")

print(
    "hlir_gpu_cross_coverage_check: pass "
    f"path={hlir_path} shared_core_pairs={len(shared_core_pairs)} "
    f"metal_coverage={len(shared_core_pairs)} ptx_coverage={len(shared_core_pairs)}"
)
PY
  if [ -x "scripts/omega/omega_hlir_gpu_cross_coverage.py" ]; then
    python3 scripts/omega/omega_hlir_gpu_cross_coverage.py --strict
  fi
  echo "HLIR_GPU_CROSS_COVERAGE_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_HLIR_LOWERING:-1}" = "1" ]; then
  echo "==> omega sprint2 gate: HLIR lowering bridge"
  HLIR_LOWERING_PATH="${OMEGA_HLIR_LOWERING_PATH:-self-hosted/hlir/lower.sio}"
  if [ ! -f "$HLIR_LOWERING_PATH" ]; then
    echo "error: HLIR lowering module missing: $HLIR_LOWERING_PATH" >&2
    exit 2
  fi
  python3 - "$HLIR_LOWERING_PATH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

if "struct HlirLowerState" not in text:
    raise SystemExit(f"{path}: missing HlirLowerState declaration")
if "loop_continue_block" not in text or "loop_break_block" not in text:
    raise SystemExit(f"{path}: missing loop control fields in HlirLowerState")

state_builder_variants = (
    ("module: HlirModuleBuilder", "current_func: HlirFunctionBuilder"),
    ("module_builder: HlirModuleBuilder", "builder: HlirFunctionBuilder"),
)
if not any(a in text and b in text for a, b in state_builder_variants):
    raise SystemExit(
        f"{path}: missing expected builder/module builder fields for HlirLowerState"
    )

scope_variants = (
    ("var_scope_keys", "var_scope_vals", "var_scope_count"),
    ("scope_stack", "scope_stack_top"),
)
if not any(all(token in text for token in variant) for variant in scope_variants):
    raise SystemExit(
        f"{path}: missing expected scope tracking fields for HlirLowerState"
    )

required_entrypoints = (
    "fn hlir_lower_module(",
    "fn hlir_lower_function(",
    "fn hlir_lower_expr(",
    "fn hlir_lower_literal(",
    "fn hlir_lower_binary(",
    "fn hlir_lower_unary(",
    "fn hlir_lower_call(",
    "fn hlir_lower_field_access(",
    "fn hlir_lower_index(",
    "fn hlir_lower_cast(",
    "fn hlir_lower_let(",
    "fn hlir_lower_var(",
    "fn hlir_lower_assign(",
    "fn hlir_lower_return(",
    "fn hlir_lower_if(",
    "fn hlir_lower_while(",
    "fn hlir_lower_match(",
    "fn hlir_lower_break(",
    "fn hlir_lower_continue(",
    "fn hlir_lower_new_block(",
    "fn hlir_lower_seal_block(",
    "fn hlir_lower_branch(",
    "fn hlir_lower_cond_branch(",
    "fn hlir_scope_push(",
    "fn hlir_scope_pop(",
    "fn hlir_scope_insert(",
    "fn hlir_scope_lookup(",
)
for marker in required_entrypoints:
    if marker not in text:
        raise SystemExit(f"{path}: missing HLIR lowering function {marker!r}")

lowered = text.lower()
if "phi" not in lowered and "ssa" not in lowered:
    raise SystemExit(f"{path}: expected SSA/phi lowering markers")

print(
    f"hlir_lowering_check: pass path={path} "
    f"entrypoints={len(required_entrypoints)}"
)
PY
  echo "HLIR_LOWERING_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_METAL_MSL_CODEGEN:-1}" = "1" ]; then
  echo "==> omega sprint2 gate: Metal MSL codegen surface"
  METAL_CODEGEN_PATH="${OMEGA_METAL_CODEGEN_PATH:-self-hosted/gpu/metal.sio}"
  if [ ! -f "$METAL_CODEGEN_PATH" ]; then
    echo "error: Metal codegen module missing: $METAL_CODEGEN_PATH" >&2
    exit 2
  fi
  python3 - "$METAL_CODEGEN_PATH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

required_symbols = (
    "struct MetalBuf",
    "fn metal_buf_new(",
    "fn metal_push_str(",
    "fn metal_push_i64(",
    "fn gpu_type_to_msl(",
    "fn metal_emit_header(",
    "fn metal_emit_kernel_signature(",
    "fn metal_emit_var_decls(",
    "fn metal_lower_ops(",
    "fn gpu_lower_to_metal(",
)
for symbol in required_symbols:
    if symbol not in text:
        raise SystemExit(f"{path}: missing Metal codegen symbol {symbol!r}")

if "#include <metal_stdlib>" not in text:
    raise SystemExit(f"{path}: missing #include <metal_stdlib> marker")
if "using namespace metal;" not in text:
    raise SystemExit(f"{path}: missing using namespace metal; marker")

print(
    f"metal_msl_codegen_check: pass path={path} "
    f"symbols={len(required_symbols)}"
)
PY
  echo "METAL_MSL_CODEGEN_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_PTX_REGALLOC_EXPANSION:-1}" = "1" ]; then
  echo "==> omega sprint2 gate: PTX regalloc expansion surface"
  PTX_ADVANCED_PATH="${OMEGA_PTX_ADVANCED_PATH:-self-hosted/gpu/ptx_advanced.sio}"
  if [ ! -f "$PTX_ADVANCED_PATH" ]; then
    echo "error: PTX advanced module missing: $PTX_ADVANCED_PATH" >&2
    exit 2
  fi
  python3 - "$PTX_ADVANCED_PATH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

required_symbols = (
    "struct PtxRegCounters",
    "fn ptx_reg_counters_new(",
    "fn ptx_alloc_pred(",
    "fn ptx_alloc_b32(",
    "fn ptx_alloc_b64(",
    "fn ptx_alloc_f32(",
    "fn ptx_alloc_f64(",
    "fn ptx_alloc_for_type(",
    "fn ptx_emit_reg_name(",
    "fn ptx_emit_typed_reg_decls(",
    "fn ptx_emit_add(",
    "fn ptx_emit_sub(",
    "fn ptx_emit_mul(",
    "fn ptx_emit_div(",
    "fn ptx_emit_rem(",
    "fn ptx_emit_fma(",
    "fn ptx_emit_ld_global(",
    "fn ptx_emit_st_global(",
    "fn ptx_emit_ld_shared(",
    "fn ptx_emit_st_shared(",
    "fn ptx_emit_ld_param(",
    "fn ptx_emit_setp_lt(",
    "fn ptx_emit_setp_le(",
    "fn ptx_emit_setp_eq(",
    "fn ptx_emit_setp_ge(",
    "fn ptx_emit_setp_gt(",
    "fn ptx_emit_setp_ne(",
    "fn ptx_emit_selp(",
    "fn ptx_emit_bra(",
    "fn ptx_emit_bra_pred(",
    "fn ptx_emit_label(",
    "fn ptx_emit_bar_sync(",
    "fn ptx_emit_exit(",
    "fn ptx_emit_ret(",
    "fn ptx_emit_cvt(",
    "fn ptx_emit_shfl_down(",
    "fn ptx_emit_shfl_up(",
    "fn ptx_emit_shfl_bfly(",
    "fn ptx_emit_vote_ballot(",
    "fn ptx_emit_bar_arrive(",
    "fn ptx_emit_sqrt(",
    "fn ptx_emit_rsqrt(",
    "fn ptx_emit_exp2(",
    "fn ptx_emit_lg2(",
    "fn ptx_emit_sin(",
    "fn ptx_emit_cos(",
    "fn ptx_emit_rcp(",
    "fn ptx_emit_abs(",
    "fn ptx_sm_version_string(",
    "fn ptx_version_for_sm(",
    "fn ptx_supports_bf16(",
    "fn ptx_supports_fp8(",
    "fn ptx_supports_tma(",
    "fn ptx_supports_cp_async_arch(",
    "fn ptx_supports_cp_async(",
    "fn ptx_supports_redux(",
    "fn ptx_supports_match(",
    "fn ptx_cp_async_size_is_valid(",
    "fn ptx_cp_async_normalize_size(",
    "fn ptx_emit_cp_async_arch(",
)
for symbol in required_symbols:
    if symbol not in text:
        raise SystemExit(f"{path}: missing PTX regalloc symbol {symbol!r}")

strict_markers = (
    ".reg .pred %p<",
    ".reg .b32 %r<",
    ".reg .b64 %rd<",
    ".reg .f32 %f<",
    ".reg .f64 %fd<",
    "ptx_supports_fp8_arch",
    "ptx_supports_tma_arch",
)
for marker in strict_markers:
    if marker not in text:
        raise SystemExit(f"{path}: missing PTX strict marker {marker!r}")

print(
    f"ptx_regalloc_expansion_check: pass path={path} "
    f"symbols={len(required_symbols)}"
)
PY
  echo "PTX_REGALLOC_EXPANSION_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_GPU_OPCODE_SMOKE:-1}" = "1" ]; then
  echo "==> omega sprint2 gate: GPU opcode runtime smoke"
  bash scripts/omega/omega_gpu_opcode_smoke.sh
  echo "GPU_OPCODE_SMOKE_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_QIR_EMITTER:-1}" = "1" ]; then
  echo "==> omega sprint4 gate: full QIR emitter"
  QIR_EMITTER_PATH="${OMEGA_QIR_FULL_EMITTER_PATH:-}"
  if [ -z "$QIR_EMITTER_PATH" ]; then
    if [ -f "hardware/rtl/qir/omega_genesis_emitter.sio" ]; then
      QIR_EMITTER_PATH="hardware/rtl/qir/omega_genesis_emitter.sio"
    else
      QIR_EMITTER_PATH="hardware/rtl/qir/omega_full_qir_emitter.sio"
    fi
  fi
  if [ ! -f "$QIR_EMITTER_PATH" ]; then
    echo "error: full QIR emitter missing: $QIR_EMITTER_PATH" >&2
    exit 2
  fi
  if [ "${OMEGA_REQUIRE_QIR_GENESIS_EMITTER:-1}" = "1" ] \
    && [ "$QIR_EMITTER_PATH" != "hardware/rtl/qir/omega_genesis_emitter.sio" ]; then
    echo "error: genesis QIR emitter required but active emitter is $QIR_EMITTER_PATH" >&2
    exit 2
  fi
  python3 - "$QIR_EMITTER_PATH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
required_symbols = (
    "omega_qir_emit_shim",
    "omega_qir_emit_quantum_controller",
    "omega_qir_emit_bundle",
    "omega_qir_full_emitter_self_check",
    "selfhost-emitter",
)
for symbol in required_symbols:
    if symbol not in text:
        raise SystemExit(f"missing required symbol in {path}: {symbol}")
if "template-direct" in text:
    raise SystemExit(f"{path}: template-direct fallback is not allowed")
print(f"qir_emitter_check: pass path={path}")
PY

  if [ -f "bootstrap/policies/policy.v2.json" ]; then
    QIR_POLICY_PATH="bootstrap/policies/policy.v2.json"
  else
    QIR_POLICY_PATH="bootstrap/policies/policy.v1.json"
  fi
  QIR_SOUC_BIN="${OMEGA_POLICY_SOUC_BIN:-$ROOT_DIR/souc}"
  set +e
  QIR_EVAL_OUTPUT="$(
    SOUNIO_POLICY_VERIFY_KEY_PATH="${OMEGA_CANONICAL_PUBKEY:-keys/bootstrap_ed25519.pub}" \
      "$QIR_SOUC_BIN" opt policy eval --policy "$QIR_POLICY_PATH" 2>&1
  )"
  QIR_EVAL_RC=$?
  set -e
  echo "$QIR_EVAL_OUTPUT"
  if [ "$QIR_EVAL_RC" -ne 0 ]; then
    echo "error: souc opt policy eval failed for QIR emitter check" >&2
    exit "$QIR_EVAL_RC"
  fi
  if ! grep -q "opt policy qir-emitter: status=pass" <<<"$QIR_EVAL_OUTPUT"; then
    echo "error: opt policy eval did not report qir-emitter pass status" >&2
    exit 2
  fi
  echo "QIR_EMITTER_PASS" | tee -a "$GATE_LOG"
  echo "QIR_GENESIS_EMITTER_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_EPI_POWER_LIVE:-1}" = "1" ]; then
  echo "==> omega sprint3 gate: QIR epistemic power .sio live module"
  QIR_SIO="hardware/rtl/qir/omega_epistemic_power.sio"
  HW_EPI_LIVE_ART="artifacts/fpga/hardware_epistemic_power_live.v1.json"
  if [ ! -f "$QIR_SIO" ]; then
    echo "error: QIR epistemic power .sio module missing: $QIR_SIO" >&2
    exit 2
  fi
  if [ ! -f "$HW_EPI_LIVE_ART" ]; then
    echo "error: hardware epistemic power live artifact missing: $HW_EPI_LIVE_ART" >&2
    exit 2
  fi
  python3 - "$QIR_SIO" "$HW_EPI_LIVE_ART" <<'PY'
import json
import sys
from pathlib import Path

sio_path = Path(sys.argv[1])
art_path = Path(sys.argv[2])
content = sio_path.read_text()
for fn in (
    "qir_epi_power_hw_q32_32",
    "qir_epi_power_variance_q32_32",
    "qir_epi_power_poll_overhead_us",
    "qir_epi_power_live_conformant_v2",
):
    if fn not in content:
        print(f"error: QIR module missing required fn {fn}", file=sys.stderr)
        sys.exit(2)
payload = json.loads(art_path.read_text())
conformant = payload.get("live_read_conformant", False)
variance_q32_32 = payload.get("hardware_epistemic_power_variance_q32_32", None)
poll_overhead_us = payload.get("poll_overhead_us", None)
if not conformant:
    print(f"error: live_read_conformant=false in {art_path}", file=sys.stderr)
    sys.exit(2)
if not isinstance(variance_q32_32, int) or variance_q32_32 < 0:
    print(
        f"error: hardware_epistemic_power_variance_q32_32 invalid in {art_path}: {variance_q32_32!r}",
        file=sys.stderr,
    )
    sys.exit(2)
if not isinstance(poll_overhead_us, int) or poll_overhead_us < 0 or poll_overhead_us >= 1000:
    print(
        f"error: poll_overhead_us must be <1000 in {art_path}: {poll_overhead_us!r}",
        file=sys.stderr,
    )
    sys.exit(2)
print(
    "QIR epi power .sio module OK "
    f"conformant={str(conformant).lower()} "
    f"variance_q32_32={variance_q32_32} "
    f"poll_overhead_us={poll_overhead_us}"
)
PY
  echo "EPI_POWER_LIVE_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_MERKLE_LANE:-1}" = "1" ]; then
  echo "==> omega sprint4 gate: K-AXI Merkle lane"
  MERKLE_SIO="hardware/rtl/kaxi/merkle_lane.sio"
  MERKLE_RTL="hardware/fpga/k_axi_merkle_lane.v"
  MERKLE_ROOT_SIO="${OMEGA_MERKLE_ROOT_SIO_PATH:-hardware/rtl/kaxi/merkle_root_lane.sio}"
  MERKLE_ROOT_RTL="${OMEGA_MERKLE_ROOT_RTL_PATH:-hardware/fpga/k_axi_merkle_root_lane.v}"
  FPGA_REPORT="artifacts/fpga/fpga_seed_report.json"
  if [ ! -f "$MERKLE_SIO" ]; then
    echo "error: missing Merkle lane .sio adapter: $MERKLE_SIO" >&2
    exit 2
  fi
  if [ ! -f "$MERKLE_RTL" ]; then
    echo "error: missing Merkle lane RTL: $MERKLE_RTL" >&2
    exit 2
  fi
  if [ ! -f "$FPGA_REPORT" ]; then
    echo "error: missing FPGA report for Merkle lane check: $FPGA_REPORT" >&2
    exit 2
  fi
  if ! rg -q "merkle_lane_root_l64" "$MERKLE_SIO"; then
    echo "error: Merkle lane .sio missing merkle_lane_root_l64" >&2
    exit 2
  fi
  if ! rg -q "merkle_lane_verify_root" "$MERKLE_SIO"; then
    echo "error: Merkle lane .sio missing merkle_lane_verify_root" >&2
    exit 2
  fi
  if ! rg -q "module k_axi_merkle_lane_core" "$MERKLE_RTL"; then
    echo "error: Merkle lane RTL missing module k_axi_merkle_lane_core" >&2
    exit 2
  fi
  if [ ! -f "$MERKLE_ROOT_SIO" ]; then
    echo "error: missing Merkle root .sio adapter: $MERKLE_ROOT_SIO" >&2
    exit 2
  fi
  if [ ! -f "$MERKLE_ROOT_RTL" ]; then
    echo "error: missing Merkle root RTL: $MERKLE_ROOT_RTL" >&2
    exit 2
  fi
  if ! rg -q "merkle_root_lane_l64" "$MERKLE_ROOT_SIO"; then
    echo "error: Merkle root .sio missing merkle_root_lane_l64" >&2
    exit 2
  fi
  if ! rg -q "merkle_root_verify_l64" "$MERKLE_ROOT_SIO"; then
    echo "error: Merkle root .sio missing merkle_root_verify_l64" >&2
    exit 2
  fi
  if ! rg -q "module k_axi_merkle_root_lane_core" "$MERKLE_ROOT_RTL"; then
    echo "error: Merkle root RTL missing module k_axi_merkle_root_lane_core" >&2
    exit 2
  fi
  python3 - "$FPGA_REPORT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("merkle_lane_present") is not True:
    raise SystemExit("merkle_lane_present must be true")
if payload.get("merkle_lane_core_rtl_present") is not True:
    raise SystemExit("merkle_lane_core_rtl_present must be true")
status = payload.get("merkle_lane_synth_status")
if status != "pass":
    raise SystemExit(f"merkle_lane_synth_status={status!r} (expected pass)")
root_status = payload.get("merkle_root_synth_status")
if root_status != "pass":
    raise SystemExit(f"merkle_root_synth_status={root_status!r} (expected pass)")
if payload.get("merkle_root_core_rtl_present") is not True:
    raise SystemExit("merkle_root_core_rtl_present must be true")
print(
    "merkle_lane_check: "
    f"present={payload.get('merkle_lane_present')} "
    f"core_rtl={payload.get('merkle_lane_core_rtl_present')} "
    f"synth={status} "
    f"root_core_rtl={payload.get('merkle_root_core_rtl_present')} "
    f"root_synth={root_status}"
)
PY
  echo "MERKLE_LANE_PASS" | tee -a "$GATE_LOG"
  echo "MERKLE_ROOT_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_BIDIR_KAXI:-1}" = "1" ]; then
  echo "==> omega sprint3 gate: bidirectional K-AXI return channel"
  FPGA_REPORT="artifacts/fpga/fpga_seed_report.json"
  BIDIR_ADAPTER_SIO="hardware/rtl/kaxi/bidirectional_return_adapter.sio"
  BIDIR_WAVE="artifacts/fpga/waveforms/tb_k_axi_bidirectional.vcd"
  if [ ! -f "$FPGA_REPORT" ]; then
    echo "error: FPGA seed report missing: $FPGA_REPORT" >&2
    exit 2
  fi
  if [ ! -f "$BIDIR_ADAPTER_SIO" ]; then
    echo "error: bidirectional K-AXI adapter missing: $BIDIR_ADAPTER_SIO" >&2
    exit 2
  fi
  if [ ! -f "$BIDIR_WAVE" ]; then
    echo "error: bidirectional K-AXI waveform missing: $BIDIR_WAVE" >&2
    exit 2
  fi
  if ! rg -q "kaxi_bidir_pack_header" "$BIDIR_ADAPTER_SIO"; then
    echo "error: bidirectional K-AXI adapter missing kaxi_bidir_pack_header" >&2
    exit 2
  fi
  if ! rg -q "kaxi_bidir_replay_inspect" "$BIDIR_ADAPTER_SIO"; then
    echo "error: bidirectional K-AXI adapter missing kaxi_bidir_replay_inspect" >&2
    exit 2
  fi
  python3 - "$FPGA_REPORT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
sim = payload.get("k_axi_return_sim_status", "")
synth = payload.get("k_axi_return_synth_status", "")
if sim != "pass":
    print(f"error: k_axi_return_sim_status={sim!r} (expected pass)", file=sys.stderr)
    sys.exit(2)
if synth != "pass":
    print(f"error: k_axi_return_synth_status={synth!r} (expected pass)", file=sys.stderr)
    sys.exit(2)
print(f"K-AXI bidirectional return: sim={sim} synth={synth}")
PY
  echo "BIDIR_KAXI_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_ACCUMULATOR_BOUNDS:-1}" = "1" ]; then
  echo "==> omega sprint3 gate: accumulator formal bounds"
  BOUNDS_SIO="tests/hardware/accumulator_bounds_test.sio"
  if [ ! -f "$BOUNDS_SIO" ]; then
    echo "error: accumulator bounds test missing: $BOUNDS_SIO" >&2
    exit 2
  fi
  SOUC_BIN="${OMEGA_POLICY_SOUC_BIN:-$ROOT_DIR/souc}"
  BOUNDS_OUT="$(PATH="$ROOT_DIR:$PATH" "$SOUC_BIN" run "$BOUNDS_SIO" 2>&1)"
  echo "$BOUNDS_OUT"
  if ! grep -q "ACCUMULATOR_BOUNDS_PASS" <<<"$BOUNDS_OUT"; then
    echo "error: accumulator_bounds_test.sio did not emit ACCUMULATOR_BOUNDS_PASS" >&2
    exit 2
  fi
  ACCUM_WAVE="artifacts/fpga/waveforms/tb_epistemic_power_accumulator.vcd"
  if [ ! -f "$ACCUM_WAVE" ]; then
    echo "error: accumulator waveform missing: $ACCUM_WAVE" >&2
    exit 2
  fi
  python3 - <<'PY'
import json
from pathlib import Path

contract_path = Path("benchmarks/independence/contract.v2.json")
if not contract_path.exists():
    contract_path = Path("benchmarks/independence/contract.v1.json")
if not contract_path.exists():
    raise SystemExit("error: missing independence contract for accumulator bounds check")

obj = json.loads(contract_path.read_text())
acc = obj.get("epistemic_power_accumulator_bound")
if not isinstance(acc, dict):
    raise SystemExit(
        f"error: {contract_path}: missing epistemic_power_accumulator_bound"
    )
formula = str(acc.get("delta_32_formula", "")).replace(" ", "")
if formula != "4*(F%8)+2*(P%16)+(Q%32)":
    raise SystemExit(
        f"error: {contract_path}: invalid delta_32_formula={formula!r}"
    )
delta = int(acc.get("delta_32_max_exclusive", 0))
if delta != 96:
    raise SystemExit(
        f"error: {contract_path}: delta_32_max_exclusive must be 96 (got {delta})"
    )
gap_lt = float(acc.get("max_gap_lt", 0.0))
if gap_lt != 3.0:
    raise SystemExit(
        f"error: {contract_path}: max_gap_lt must be 3.0 (got {gap_lt})"
    )
print(
    f"accumulator_contract_bound_check: pass contract={contract_path} "
    f"delta_32_max_exclusive={delta} max_gap_lt={gap_lt}"
)
PY
  echo "ACCUMULATOR_BOUNDS_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_INTERNED_CHEMICAL_PROVENANCE:-1}" = "1" ]; then
  echo "==> omega sprint4 gate: interned chemical provenance"
  INTERNER_SIO="self-hosted/intern.sio"
  if [ ! -f "$INTERNER_SIO" ]; then
    echo "error: interner module missing: $INTERNER_SIO" >&2
    exit 2
  fi
  python3 - "$INTERNER_SIO" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
required_symbols = (
    "variance_q32_32",
    "provenance_root_l64",
    "confidence_beta",
    "intern_merkle_root_lane_l64",
    "interner_epi_power_delta_32",
    "interner_chemical_provenance_self_check",
)
for symbol in required_symbols:
    if symbol not in text:
        raise SystemExit(f"{path}: missing required symbol {symbol!r}")

compact = text.replace(" ", "")
formula = "frac_f*4+frac_p*2+frac_q"
if formula not in compact:
    raise SystemExit(f"{path}: missing accumulator delta_32 formula fragment {formula!r}")

print(f"interned_chemical_provenance_check: pass path={path}")
PY
  echo "INTERNED_CHEMICAL_PROVENANCE_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_INTERNED_CHEMICAL_PROOF:-1}" = "1" ]; then
  echo "==> omega sprint4 gate: interned chemical Lean proof linkage"
  python3 - <<'PY'
from pathlib import Path

proof_path = Path("formal/omega_mathlib/OmegaMathlib/InternedChemicalGraph.lean")
root_path = Path("formal/omega_mathlib/OmegaMathlib.lean")
lakefile_path = Path("formal/omega_mathlib/lakefile.lean")

for path in (proof_path, root_path, lakefile_path):
    if not path.exists():
        raise SystemExit(f"{path}: required file missing")

proof_text = proof_path.read_text()
for symbol in (
    "import OmegaMathlib.AccumulatorBounds",
    "theorem intern_delta32_lt_96",
    "theorem intern_preserves_accumulator_bound",
):
    if symbol not in proof_text:
        raise SystemExit(f"{proof_path}: missing required symbol {symbol!r}")

root_text = root_path.read_text()
if "import OmegaMathlib.InternedChemicalGraph" not in root_text:
    raise SystemExit(
        f"{root_path}: missing import OmegaMathlib.InternedChemicalGraph"
    )

lakefile_text = lakefile_path.read_text()
if "`OmegaMathlib.InternedChemicalGraph" not in lakefile_text:
    raise SystemExit(
        f"{lakefile_path}: missing root `OmegaMathlib.InternedChemicalGraph"
    )

print("interned_chemical_proof_check: pass")
PY
  echo "INTERNED_CHEMICAL_PROOF_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_RESOURCE_TREND:-1}" = "1" ]; then
  echo "==> omega sprint4 gate: hardware resource trend"
  RESOURCE_TREND_PATH="${OMEGA_HW_RESOURCE_TREND_PATH:-artifacts/fpga/hardware_resource_trend.v2.json}"
  if [ ! -f "$RESOURCE_TREND_PATH" ]; then
    echo "error: hardware resource trend artifact missing: $RESOURCE_TREND_PATH" >&2
    exit 2
  fi
  python3 - "$RESOURCE_TREND_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
runs = payload.get("runs")
if not isinstance(runs, list) or not runs:
    raise SystemExit(f"{path}: runs missing or empty")
last = runs[-1]
if not isinstance(last, dict):
    raise SystemExit(f"{path}: latest run must be object")
modules = last.get("modules")
if not isinstance(modules, dict) or len(modules) < 3:
    raise SystemExit(f"{path}: latest run modules missing or too small")
status = payload.get("last_status")
if status not in ("pass", "bootstrap"):
    raise SystemExit(f"{path}: last_status must be pass/bootstrap, got {status!r}")
max_drift = payload.get("last_max_relative_drift")
threshold = payload.get("drift_threshold", 0.05)
if not isinstance(max_drift, (int, float)):
    raise SystemExit(f"{path}: last_max_relative_drift missing")
if not isinstance(threshold, (int, float)):
    raise SystemExit(f"{path}: drift_threshold missing")
if float(max_drift) > float(threshold):
    raise SystemExit(
        f"{path}: max drift {float(max_drift):.6f} exceeds threshold {float(threshold):.6f}"
    )
print(
    f"resource_trend_check: status={status} "
    f"max_relative_drift={float(max_drift):.6f} "
    f"threshold={float(threshold):.6f} "
    f"modules={len(modules)}"
)
PY
  echo "RESOURCE_TREND_GATE_PASS" | tee -a "$GATE_LOG"
  echo "RESOURCE_ZERO_DRIFT_PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_GENESIS_MANIFEST:-1}" = "1" ]; then
  echo "==> omega sprint4 gate: genesis manifest release"
  GENESIS_ARGS=(
    --gate-log "$GATE_LOG"
    --canonical-privkey "${OMEGA_CANONICAL_PRIVKEY:-keys/bootstrap_ed25519}"
    --canonical-pubkey "${OMEGA_CANONICAL_PUBKEY:-keys/bootstrap_ed25519.pub}"
    --canonical-timestamp "${OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP_PATH:-keys/bootstrap_ed25519.created_at}"
    --out "${OMEGA_GENESIS_MANIFEST_OUT:-artifacts/omega/omega_genesis.v1.0.json}"
  )
  if [ "${OMEGA_REQUIRE_GENESIS_MANIFEST_STRICT:-1}" = "1" ]; then
    GENESIS_ARGS+=(--strict)
  fi
  python3 scripts/omega/omega_genesis_manifest.py "${GENESIS_ARGS[@]}"
  echo "OMEGA_GENESIS_V1_RELEASE_PASS" | tee -a "$GATE_LOG"
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
    RL_POLICY_STATUS_SCRIPT="${OMEGA_POLICY_STATUS_SCRIPT:-scripts/omega/omega_policy_status.sh}"
    RL_CANONICAL_ENV_PATH="${OMEGA_CANONICAL_ENV_OUT:-artifacts/omega/canonical_key.env}"
    RL_SOUC_BIN="${OMEGA_POLICY_SOUC_BIN:-$ROOT_DIR/souc}"
    PATH="$ROOT_DIR:$PATH" \
    SOUNIO_RL_READINESS_EVIDENCE_PATH="${OMEGA_RL_READINESS_EVIDENCE_OUT:-bootstrap/policies/rl_readiness.evidence.json}" \
      "$RL_POLICY_STATUS_SCRIPT" \
        --policy "$RL_POLICY_PATH" \
        --souc "$RL_SOUC_BIN" \
        --corpus "${OMEGA_POLICY_TRAIN_CORPUS:-benchmarks/independence}" \
        --canonical-env "$RL_CANONICAL_ENV_PATH" \
        --smoke-env "$RL_POLICY_SMOKE_ENV_PATH" \
        --smoke-out "$RL_POLICY_SMOKE_OUTPUT" | tee "$RL_STATUS_LOG"
    if ! rg -q "opt policy readiness: pass" "$RL_STATUS_LOG"; then
      echo "error: RL readiness status did not report pass" >&2
      exit 2
    fi
    if ! rg -q "signature=verified \(canonical in-place\)" "$RL_STATUS_LOG"; then
      echo "error: RL readiness status did not verify canonical in-place signature" >&2
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

if [ "${OMEGA_REQUIRE_PINNED_DIGEST:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: pinned digest sign+verify"
  PIN_FREEZE_PATH="${OMEGA_BASELINE_FREEZE_OUT:-artifacts/omega/baseline_freeze.v1.json}"
  if [ ! -f "$PIN_FREEZE_PATH" ]; then
    echo "error: pinned digest gate requires baseline freeze artifact: $PIN_FREEZE_PATH" >&2
    exit 2
  fi
  PIN_DIGEST="$(python3 - "$PIN_FREEZE_PATH" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
value = payload.get("policy_pinned_digest_sha256", "")
print(value if isinstance(value, str) else "")
PY
)"
  if [ -n "$PIN_DIGEST" ]; then
    if ! [[ "$PIN_DIGEST" =~ ^[0-9a-fA-F]{64}$ ]]; then
      echo "error: invalid pinned digest from freeze artifact $PIN_FREEZE_PATH: $PIN_DIGEST" >&2
      exit 2
    fi
    PIN_DIGEST="${PIN_DIGEST,,}"
  fi

  if [ -f "bootstrap/policies/policy.v2.json" ]; then
    PIN_POLICY_PATH="bootstrap/policies/policy.v2.json"
  else
    PIN_POLICY_PATH="bootstrap/policies/policy.v1.json"
  fi
  PIN_CANONICAL_ENV_PATH="${OMEGA_CANONICAL_ENV_OUT:-artifacts/omega/canonical_key.env}"
  PIN_POLICY_SIGN_SCRIPT="${OMEGA_CANONICAL_POLICY_SIGN_SCRIPT:-scripts/omega_canonical_policy_sign.sh}"
  PIN_SOUC_BIN="${OMEGA_POLICY_SOUC_BIN:-$ROOT_DIR/souc}"
  if [ ! -x "$PIN_POLICY_SIGN_SCRIPT" ]; then
    echo "error: missing canonical policy sign script: $PIN_POLICY_SIGN_SCRIPT" >&2
    exit 2
  fi

  PIN_SIGN_ARGS=(
    --policy "$PIN_POLICY_PATH"
    --out "$PIN_POLICY_PATH"
    --souc "$PIN_SOUC_BIN"
    --canonical-env "$PIN_CANONICAL_ENV_PATH"
  )
  if [ -n "$PIN_DIGEST" ]; then
    PIN_SIGN_ARGS+=(--pin-digest "$PIN_DIGEST")
  fi
  OMEGA_POLICY_PIN_FROM_FREEZE=1 \
  OMEGA_POLICY_PIN_DIGEST="$PIN_DIGEST" \
  OMEGA_POLICY_PIN_FREEZE_PATH="$PIN_FREEZE_PATH" \
    "$PIN_POLICY_SIGN_SCRIPT" "${PIN_SIGN_ARGS[@]}"

  # Negative check: signing with an incorrect pin digest must fail.
  BAD_PIN_DIGEST="0000000000000000000000000000000000000000000000000000000000000000"
  if [ -n "$PIN_DIGEST" ]; then
    BAD_PIN_DIGEST="$(python3 - "$PIN_DIGEST" <<'PY'
import sys
value = sys.argv[1].strip().lower()
if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
    print("0" * 64)
    raise SystemExit(0)
first = value[0]
replacement = "0" if first != "0" else "1"
print(replacement + value[1:])
PY
)"
  fi

  NEG_POLICY_COPY="$(mktemp "${TMPDIR:-/tmp}/omega_pinned_digest_negative.XXXXXX.json")"
  cp "$PIN_POLICY_PATH" "$NEG_POLICY_COPY"
  set +e
  OMEGA_CANONICAL_PRIVKEY="${OMEGA_CANONICAL_PRIVKEY:-keys/bootstrap_ed25519}" \
  OMEGA_CANONICAL_PUBKEY="${OMEGA_CANONICAL_PUBKEY:-keys/bootstrap_ed25519.pub}" \
    "$PIN_SOUC_BIN" opt policy sign \
      --policy "$NEG_POLICY_COPY" \
      --out "$NEG_POLICY_COPY" \
      --pin-digest "$BAD_PIN_DIGEST" >/dev/null 2>&1
  NEG_PIN_RC=$?
  set -e
  rm -f "$NEG_POLICY_COPY"
  if [ "$NEG_PIN_RC" -eq 0 ]; then
    echo "error: negative pin digest test failed (sign unexpectedly succeeded with bad pin)" >&2
    exit 2
  fi

  PIN_STATUS_LOG="${OMEGA_PINNED_DIGEST_STATUS_LOG:-artifacts/omega_pinned_digest_status.log}"
  set +e
  PIN_STATUS_OUTPUT="$(
    SOUNIO_POLICY_VERIFY_KEY_PATH="${OMEGA_CANONICAL_PUBKEY:-keys/bootstrap_ed25519.pub}" \
      "$PIN_SOUC_BIN" opt policy status --policy "$PIN_POLICY_PATH" 2>&1
  )"
  PIN_STATUS_RC=$?
  set -e
  echo "$PIN_STATUS_OUTPUT" | tee "$PIN_STATUS_LOG"
  if [ "$PIN_STATUS_RC" -ne 0 ]; then
    echo "error: pinned digest status command failed: policy=$PIN_POLICY_PATH" >&2
    exit "$PIN_STATUS_RC"
  fi
  if ! rg -q "signature=verified" "$PIN_STATUS_LOG"; then
    echo "error: pinned digest status did not report signature=verified" >&2
    exit 2
  fi
  if ! rg -q "opt policy canonical: signature=verified" "$PIN_STATUS_LOG"; then
    echo "error: pinned digest status did not verify canonical signature" >&2
    exit 2
  fi

  if ! rg -q "opt policy pinned: status=verified" "$PIN_STATUS_LOG"; then
    echo "error: pinned digest status was not verified" >&2
    exit 2
  fi
  if [ -n "$PIN_DIGEST" ] && ! rg -q "pinned=$PIN_DIGEST" "$PIN_STATUS_LOG"; then
    echo "error: pinned digest status does not match freeze policy pin digest" >&2
    exit 2
  fi

  echo "PINNED_DIGEST_PASS" | tee -a "$GATE_LOG"
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
    --canonical-privkey "${OMEGA_CANONICAL_PRIVKEY:-}"
    --canonical-pubkey "${OMEGA_CANONICAL_PUBKEY:-keys/bootstrap_ed25519.pub}"
    --canonical-timestamp "${OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP_PATH:-keys/bootstrap_ed25519.created_at}"
    --policy "${RL_POLICY_PATH:-bootstrap/policies/policy.v2.json}"
    --baseline-freeze "${OMEGA_BASELINE_FREEZE_OUT:-artifacts/omega/baseline_freeze.v1.json}"
    --resource-trend "${OMEGA_HW_RESOURCE_TREND_PATH:-artifacts/fpga/hardware_resource_trend.v2.json}"
    --qir-emitter "${OMEGA_QIR_FULL_EMITTER_PATH:-hardware/rtl/qir/omega_genesis_emitter.sio}"
    --merkle-sio "${OMEGA_MERKLE_ROOT_SIO_PATH:-hardware/rtl/kaxi/merkle_root_lane.sio}"
    --merkle-rtl "${OMEGA_MERKLE_ROOT_RTL_PATH:-hardware/fpga/k_axi_merkle_root_lane.v}"
    --genesis-manifest "${OMEGA_GENESIS_MANIFEST_OUT:-artifacts/omega/omega_genesis.v1.0.json}"
    --key-id "${OMEGA_GOVERNANCE_ATTEST_KEY_ID:-canonical-bootstrap-ed25519}"
    --require-signature
  )
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
  OMEGA_REQUIRE_BIDIR_KAXI="${OMEGA_REQUIRE_BIDIR_KAXI:-1}"
  OMEGA_REQUIRE_ACCUMULATOR_BOUNDS="${OMEGA_REQUIRE_ACCUMULATOR_BOUNDS:-1}"
  OMEGA_REQUIRE_EPI_POWER_LIVE="${OMEGA_REQUIRE_EPI_POWER_LIVE:-1}"
  OMEGA_REQUIRE_QIR_EMITTER="${OMEGA_REQUIRE_QIR_EMITTER:-1}"
  OMEGA_REQUIRE_MERKLE_LANE="${OMEGA_REQUIRE_MERKLE_LANE:-1}"
  OMEGA_REQUIRE_MERKLE_ROOT="${OMEGA_REQUIRE_MERKLE_ROOT:-1}"
  OMEGA_REQUIRE_GENESIS_MANIFEST="${OMEGA_REQUIRE_GENESIS_MANIFEST:-1}"
  OMEGA_REQUIRE_RESOURCE_TREND="${OMEGA_REQUIRE_RESOURCE_TREND:-1}"
  OMEGA_REQUIRE_INTERNED_CHEMICAL_PROVENANCE="${OMEGA_REQUIRE_INTERNED_CHEMICAL_PROVENANCE:-1}"
  OMEGA_REQUIRE_INTERNED_CHEMICAL_PROOF="${OMEGA_REQUIRE_INTERNED_CHEMICAL_PROOF:-1}"
  OMEGA_REQUIRE_DATA_STRUCTURES="${OMEGA_REQUIRE_DATA_STRUCTURES:-1}"
  OMEGA_REQUIRE_GPU_IR_EXPANSION="${OMEGA_REQUIRE_GPU_IR_EXPANSION:-1}"
  OMEGA_REQUIRE_HLIR_GPU_CROSS_COVERAGE="${OMEGA_REQUIRE_HLIR_GPU_CROSS_COVERAGE:-1}"
  OMEGA_REQUIRE_HLIR_LOWERING="${OMEGA_REQUIRE_HLIR_LOWERING:-1}"
  OMEGA_REQUIRE_METAL_MSL_CODEGEN="${OMEGA_REQUIRE_METAL_MSL_CODEGEN:-1}"
  OMEGA_REQUIRE_PTX_REGALLOC_EXPANSION="${OMEGA_REQUIRE_PTX_REGALLOC_EXPANSION:-1}"
  OMEGA_REQUIRE_SOUC_RELEASE_PROVENANCE="${OMEGA_REQUIRE_SOUC_RELEASE_PROVENANCE:-0}"
  OMEGA_REQUIRE_REPO_HARD_NO_RUST="${OMEGA_REQUIRE_REPO_HARD_NO_RUST:-0}"
  OMEGA_REQUIRE_PARALLEL_SELFHOST_CUTOVER="${OMEGA_REQUIRE_PARALLEL_SELFHOST_CUTOVER:-0}"
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
  OMEGA_REQUIRE_BIDIR_KAXI="$OMEGA_REQUIRE_BIDIR_KAXI" \
  OMEGA_REQUIRE_ACCUMULATOR_BOUNDS="$OMEGA_REQUIRE_ACCUMULATOR_BOUNDS" \
  OMEGA_REQUIRE_EPI_POWER_LIVE="$OMEGA_REQUIRE_EPI_POWER_LIVE" \
  OMEGA_REQUIRE_QIR_EMITTER="$OMEGA_REQUIRE_QIR_EMITTER" \
  OMEGA_REQUIRE_MERKLE_LANE="$OMEGA_REQUIRE_MERKLE_LANE" \
  OMEGA_REQUIRE_MERKLE_ROOT="$OMEGA_REQUIRE_MERKLE_ROOT" \
  OMEGA_REQUIRE_GENESIS_MANIFEST="$OMEGA_REQUIRE_GENESIS_MANIFEST" \
  OMEGA_REQUIRE_RESOURCE_TREND="$OMEGA_REQUIRE_RESOURCE_TREND" \
  OMEGA_REQUIRE_INTERNED_CHEMICAL_PROVENANCE="$OMEGA_REQUIRE_INTERNED_CHEMICAL_PROVENANCE" \
  OMEGA_REQUIRE_INTERNED_CHEMICAL_PROOF="$OMEGA_REQUIRE_INTERNED_CHEMICAL_PROOF" \
  OMEGA_REQUIRE_DATA_STRUCTURES="$OMEGA_REQUIRE_DATA_STRUCTURES" \
  OMEGA_REQUIRE_GPU_IR_EXPANSION="$OMEGA_REQUIRE_GPU_IR_EXPANSION" \
  OMEGA_REQUIRE_HLIR_GPU_CROSS_COVERAGE="$OMEGA_REQUIRE_HLIR_GPU_CROSS_COVERAGE" \
  OMEGA_REQUIRE_HLIR_LOWERING="$OMEGA_REQUIRE_HLIR_LOWERING" \
  OMEGA_REQUIRE_METAL_MSL_CODEGEN="$OMEGA_REQUIRE_METAL_MSL_CODEGEN" \
  OMEGA_REQUIRE_PTX_REGALLOC_EXPANSION="$OMEGA_REQUIRE_PTX_REGALLOC_EXPANSION" \
  OMEGA_REQUIRE_SOUC_RELEASE_PROVENANCE="$OMEGA_REQUIRE_SOUC_RELEASE_PROVENANCE" \
  OMEGA_REQUIRE_REPO_HARD_NO_RUST="$OMEGA_REQUIRE_REPO_HARD_NO_RUST" \
  OMEGA_REQUIRE_PARALLEL_SELFHOST_CUTOVER="$OMEGA_REQUIRE_PARALLEL_SELFHOST_CUTOVER" \
    python3 scripts/omega/omega_hardware_telemetry_regression.py "${HW_TELEMETRY_ARGS[@]}"
  echo "HW_TELEMETRY_REGRESSION_PASS" | tee -a "$GATE_LOG"
else
  echo "omega sprint1 gate: hardware telemetry regression disabled (OMEGA_REQUIRE_HW_TELEMETRY_REGRESSION=0)"
fi

if [ "${OMEGA_REQUIRE_SOUC_RELEASE_PROVENANCE:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: souc release provenance"
  PROVENANCE_PATH="${OMEGA_SOUC_PROVENANCE_OUT:-artifacts/omega/souc_release_provenance.v1.json}"
  if [ ! -f "$PROVENANCE_PATH" ]; then
    echo "error: missing souc release provenance artifact: $PROVENANCE_PATH" >&2
    exit 2
  fi
  python3 - "$PROVENANCE_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
obj = json.loads(path.read_text())
if obj.get("schema") != "sounio.omega.souc-release-provenance.v1":
    raise SystemExit(f"{path}: invalid schema {obj.get('schema')!r}")
if obj.get("status") != "pass":
    raise SystemExit(f"{path}: status must be pass (got {obj.get('status')!r})")
sha = obj.get("sha256", {})
if sha.get("match") is not True:
    raise SystemExit(f"{path}: sha256.match must be true")
sig = obj.get("signature", {})
if sig.get("verified") is not True:
    raise SystemExit(f"{path}: signature.verified must be true")
print(f"souc_release_provenance_check: pass path={path}")
PY
  echo "SOUC_RELEASE_PROVENANCE_PASS" | tee -a "$GATE_LOG"
fi

OMEGA_REQUIRE_REPO_HARD_NO_RUST="${OMEGA_REQUIRE_REPO_HARD_NO_RUST:-${OMEGA_REQUIRE_NO_RUST:-1}}"
if [ "${OMEGA_REQUIRE_REPO_HARD_NO_RUST}" = "1" ]; then
  echo "==> omega sprint1 gate: no-rust execution surface"
  if [ ! -x "scripts/omega/omega_no_rust_execution_surface.sh" ]; then
    echo "error: missing no-rust execution surface script" >&2
    exit 2
  fi
  scripts/omega/omega_no_rust_execution_surface.sh
  echo "REPO_HARD_NO_RUST_PASS" | tee -a "$GATE_LOG"

  echo "==> omega sprint1 gate: no-rust mode guard"
  changed_rust=()
  if [ -d "crates" ]; then
    mapfile -t changed_rust < <(find crates -type f -name '*.rs' -newer "$NO_RUST_MARKER" | sort)
  fi
  if [ "${#changed_rust[@]}" -ne 0 ]; then
    echo "error: no-rust mode violated; Rust files modified during gate run:" >&2
    printf '  %s\n' "${changed_rust[@]}" >&2
    exit 2
  fi
  echo "No-Rust mode PASS" | tee -a "$GATE_LOG"
fi

if [ "${OMEGA_REQUIRE_DRIVER_HARNESS_STATUS:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: driver harness readiness"
  DRIVER_HARNESS_CACHE_DIR="${OMEGA_DRIVER_HARNESS_CACHE_DIR:-artifacts/omega/driver_harness_cache}"
  DRIVER_HARNESS_STATUS_LOG="${OMEGA_DRIVER_HARNESS_STATUS_LOG:-artifacts/omega/driver_harness_status.log}"
  DRIVER_HARNESS_TIMEOUT_SECS="${OMEGA_DRIVER_HARNESS_TIMEOUT_SECS:-120}"
  mkdir -p "$DRIVER_HARNESS_CACHE_DIR" "$(dirname "$DRIVER_HARNESS_STATUS_LOG")"

  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${DRIVER_HARNESS_TIMEOUT_SECS}s" env \
      SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR="$DRIVER_HARNESS_CACHE_DIR" \
      "$OMEGA_POLICY_SOUC_BIN" run examples/minimal.sio --use-sounio-compiler --check-only \
      >"$DRIVER_HARNESS_STATUS_LOG" 2>&1
  else
    env \
      SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR="$DRIVER_HARNESS_CACHE_DIR" \
      "$OMEGA_POLICY_SOUC_BIN" run examples/minimal.sio --use-sounio-compiler --check-only \
      >"$DRIVER_HARNESS_STATUS_LOG" 2>&1
  fi
  DRIVER_HARNESS_RC=$?
  set -e

  if [ "$DRIVER_HARNESS_RC" -eq 0 ] && ls "$DRIVER_HARNESS_CACHE_DIR"/driver_harness_*.sobc >/dev/null 2>&1; then
    echo "SELFHOST_DRIVER_HARNESS_STATUS status=ready rc=$DRIVER_HARNESS_RC cache_dir=$DRIVER_HARNESS_CACHE_DIR log=$DRIVER_HARNESS_STATUS_LOG" | tee -a "$GATE_LOG"
    echo "DRIVER_HARNESS_READY_PASS" | tee -a "$GATE_LOG"
  elif rg -n "SELFHOST_DRIVER_HARNESS_UNAVAILABLE" "$DRIVER_HARNESS_STATUS_LOG" >/dev/null 2>&1; then
    echo "SELFHOST_DRIVER_HARNESS_STATUS status=unavailable rc=$DRIVER_HARNESS_RC reason=SELFHOST_DRIVER_HARNESS_UNAVAILABLE cache_dir=$DRIVER_HARNESS_CACHE_DIR log=$DRIVER_HARNESS_STATUS_LOG" | tee -a "$GATE_LOG"
    if [ "${OMEGA_REQUIRE_DRIVER_HARNESS_AVAILABLE:-0}" = "1" ]; then
      echo "error: driver harness unavailable under strict requirement" >&2
      exit 2
    fi
  else
    echo "SELFHOST_DRIVER_HARNESS_STATUS status=unknown rc=$DRIVER_HARNESS_RC cache_dir=$DRIVER_HARNESS_CACHE_DIR log=$DRIVER_HARNESS_STATUS_LOG" | tee -a "$GATE_LOG"
    if [ "${OMEGA_REQUIRE_DRIVER_HARNESS_AVAILABLE:-0}" = "1" ]; then
      echo "error: driver harness status unknown under strict requirement" >&2
      exit 2
    fi
  fi
fi

if [ "${OMEGA_REQUIRE_PARALLEL_SELFHOST_CUTOVER:-1}" = "1" ]; then
  echo "==> omega sprint1 gate: parallel selfhost cutover status"
  python3 scripts/omega/omega_parallel_cutover_status.py --gate-log "$GATE_LOG"
  echo "PARALLEL_SELFHOST_CUTOVER_PASS" | tee -a "$GATE_LOG"
fi

echo "OMEGA_SPRINT1_GATE_PASS"
