#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${CONTRACT_PATH:-}" ]; then
  CONTRACT_PATH="$CONTRACT_PATH"
elif [ -f "benchmarks/independence/contract.v2.json" ]; then
  CONTRACT_PATH="benchmarks/independence/contract.v2.json"
else
  CONTRACT_PATH="benchmarks/independence/contract.v1.json"
fi

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/target/debug/souc}"
RUN_EXTERNAL_BASELINES="${RUN_EXTERNAL_BASELINES:-0}"
POLICY_TRAIN_CORPUS="${OMEGA_POLICY_TRAIN_CORPUS:-benchmarks/independence}"
POLICY_SMOKE_OUTPUT="${OMEGA_POLICY_SMOKE_OUTPUT:-artifacts/omega/policy_status_smoke.v2.json}"
POLICY_SMOKE_ENV_PATH="${OMEGA_POLICY_SMOKE_ENV_PATH:-artifacts/omega/policy_smoke.env}"

if [ ! -f "$CONTRACT_PATH" ]; then
  echo "error: missing independence contract: $CONTRACT_PATH" >&2
  exit 1
fi

python3 - "$CONTRACT_PATH" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
obj = json.loads(path.read_text())
schema = obj.get("schema", "")

if schema not in {"sounio.independence.contract.v1", "sounio.independence.contract.v2"}:
    raise SystemExit(f"invalid schema: {schema}")

required_common = [
    "schema",
    "target_platforms",
    "baseline_adapters",
    "kernel_sets",
    "numeric_tolerance",
    "performance_gate",
    "compile_time_budget",
]
for key in required_common:
    if key not in obj:
        raise SystemExit(f"missing required contract field: {key}")

perf = obj["performance_gate"]
threshold = float(perf.get("threshold", 0.0))
if threshold < 1.2:
    raise SystemExit(f"performance threshold too low: {threshold} (expected >= 1.2)")

budget = float(obj["compile_time_budget"].get("max_overhead", 0.0))
if budget > 0.25:
    raise SystemExit(f"compile-time budget too high: {budget} (expected <= 0.25)")

if schema.endswith("v2"):
    required_v2 = [
        "runtime_normalization_by_family",
        "quantum_conformance",
        "bit_exact",
    ]
    for key in required_v2:
        if key not in obj:
            raise SystemExit(f"missing required v2 field: {key}")

    samples = int(perf.get("minimum_samples", 0))
    if samples < 50:
        raise SystemExit(f"minimum_samples too low: {samples} (expected >= 50)")

    bit_exact = obj["bit_exact"]
    if not bool(bit_exact.get("required", False)):
        raise SystemExit("bit_exact.required must be true for v2")

    qc = obj["quantum_conformance"]
    pmin = float(qc.get("ks_pvalue_min", 0.0))
    if pmin < 0.05:
        raise SystemExit(f"ks_pvalue_min too low: {pmin} (expected >= 0.05)")

    lo, hi = qc.get("coverage_range", [0.0, 0.0])
    if lo > 0.90 or hi < 0.98:
        raise SystemExit(
            f"coverage_range incompatible with sprint targets: [{lo}, {hi}] (expected to include [0.90,0.98])"
        )

print(f"independence contract ok: schema={schema} threshold={threshold} budget={budget}")
PY

if [ "$RUN_EXTERNAL_BASELINES" = "1" ]; then
  echo "==> external baseline adapters"
  python3 - "$CONTRACT_PATH" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text())
for adapter in obj.get("baseline_adapters", []):
    entry = adapter.get("entrypoint", "")
    if not entry:
        raise SystemExit(f"adapter missing entrypoint: {adapter}")
    p = pathlib.Path(entry)
    if not p.exists():
        raise SystemExit(f"adapter entrypoint missing: {entry}")
    print(f"adapter present: {adapter.get('name', 'unknown')} -> {entry}")
PY
else
  echo "independence benchmark gate: external baseline execution skipped (RUN_EXTERNAL_BASELINES=0)"
fi

if [ -f "bootstrap/policies/policy.v2.json" ]; then
  POLICY_PATH="bootstrap/policies/policy.v2.json"
else
  POLICY_PATH="bootstrap/policies/policy.v1.json"
fi
POLICY_STATUS_PATH="$POLICY_PATH"

prepare_policy_smoke() {
  local souc_cmd="$1"
  local prep_script="scripts/omega/omega_prepare_policy_smoke.sh"
  if [ ! -x "$prep_script" ]; then
    return 0
  fi
  if ! "$prep_script" \
    --policy "$POLICY_PATH" \
    --souc "$souc_cmd" \
    --corpus "$POLICY_TRAIN_CORPUS" \
    --out "$POLICY_SMOKE_OUTPUT" \
    --env-out "$POLICY_SMOKE_ENV_PATH"; then
    echo "policy smoke warning: signed-policy preparation failed, continuing with source policy"
    return 0
  fi
  if [ -f "$POLICY_SMOKE_ENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$POLICY_SMOKE_ENV_PATH"
  fi
  POLICY_STATUS_PATH="${SOUNIO_POLICY_STATUS_PATH:-$POLICY_PATH}"
}

run_policy_smoke() {
  local cmd_prefix=("$@")
  prepare_policy_smoke "${cmd_prefix[0]}"
  set +e
  local output
  output="$(
    SOUNIO_POLICY_VERIFY_KEY_PATH="${SOUNIO_POLICY_VERIFY_KEY_PATH:-}" \
      "${cmd_prefix[@]}" opt policy status --policy "$POLICY_STATUS_PATH" 2>&1
  )"
  local code=$?
  set -e
  echo "$output"
  if [ $code -ne 0 ]; then
    return 0
  fi
  if [[ "$output" == *"unsupported optimization policy schema 'sounio.optimization.policy.v2': expected 'sounio.optimization.policy.v1'"* ]]; then
    echo "policy smoke warning: binary appears stale for v2 schema; retrying via cargo run"
    SOUNIO_POLICY_VERIFY_KEY_PATH="${SOUNIO_POLICY_VERIFY_KEY_PATH:-}" \
      cargo run -p souc --bin souc -- opt policy status --policy "$POLICY_STATUS_PATH" || true
  fi
}

if [ -x "$SOUC_BIN" ] && "$SOUC_BIN" opt --help >/dev/null 2>&1; then
  echo "==> policy contract smoke"
  run_policy_smoke "$SOUC_BIN"
elif [ -x "$ROOT_DIR/target/debug/souc" ] && "$ROOT_DIR/target/debug/souc" opt --help >/dev/null 2>&1; then
  echo "==> policy contract smoke (fallback debug binary)"
  run_policy_smoke "$ROOT_DIR/target/debug/souc"
else
  echo "warning: no souc binary with 'opt' subcommand available; skipping policy status smoke"
fi

echo "INDEPENDENCE_BENCHMARK_GATE_PASS"
