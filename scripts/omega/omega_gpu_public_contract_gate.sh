#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: required dependency missing: jq" >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: required dependency missing: python3" >&2
  exit 2
fi

MODE="${OMEGA_GPU_PUBLIC_CONTRACT_MODE:-auto}"
OUT_JSON="${OMEGA_GPU_PUBLIC_CONTRACT_OUT:-$ROOT_DIR/artifacts/omega/gpu_public_contract.v1.json}"
LOG_PATH="${OMEGA_GPU_PUBLIC_CONTRACT_LOG:-$ROOT_DIR/artifacts/omega/gpu_public_contract.log}"
GPU_FIXTURE="${OMEGA_GPU_PUBLIC_CONTRACT_FIXTURE:-$ROOT_DIR/scripts/fixtures/gpu_minimal.sio}"
GPU_SOUC_HINT="${OMEGA_GPU_PUBLIC_CONTRACT_SOUC_BIN:-}"
CODEGEN_PARITY_JSON="${OMEGA_GPU_CODEGEN_PARITY_JSON:-$ROOT_DIR/artifacts/omega/gpu_codegen_parity.v1.json}"
GPU_BINARY_ATTEST_JSON="${OMEGA_GPU_BINARY_ATTEST_JSON:-$ROOT_DIR/artifacts/omega/gpu_binary_attestation.v1.json}"
GPU_RUNTIME_GATE_JSON="${OMEGA_GPU_RUNTIME_GATE_JSON:-$ROOT_DIR/artifacts/omega/gpu_runtime_attest_gate.v1.json}"

case "$MODE" in
  auto|required|off) ;;
  *)
    echo "error: OMEGA_GPU_PUBLIC_CONTRACT_MODE must be auto|required|off (got '$MODE')" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$LOG_PATH")"
: >"$LOG_PATH"

to_rel() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${path#$ROOT_DIR/}"
  else
    printf '%s' "$path"
  fi
}

emit_json() {
  local status="$1"
  local reason="$2"
  local blockers_json="$3"
  local checks_json="$4"
  local profile_json="$5"
  python3 - "$OUT_JSON" "$MODE" "$status" "$reason" "$blockers_json" "$checks_json" "$profile_json" \
    "$LOG_PATH" "$CODEGEN_PARITY_JSON" "$GPU_BINARY_ATTEST_JSON" "$GPU_RUNTIME_GATE_JSON" "$ROOT_DIR" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

out_json = Path(sys.argv[1])
mode = sys.argv[2]
status = sys.argv[3]
reason = sys.argv[4]
blockers = json.loads(sys.argv[5])
checks = json.loads(sys.argv[6])
profile = json.loads(sys.argv[7])
log_path = Path(sys.argv[8])
parity_path = Path(sys.argv[9])
binary_path = Path(sys.argv[10])
runtime_path = Path(sys.argv[11])
root = Path(sys.argv[12]).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


parity = load_json(parity_path) or {}
binary = load_json(binary_path) or {}
runtime = load_json(runtime_path) or {}

# Surface CUDA toolchain provenance from whichever sub-artifact captured it
# (runtime ran the kernels; parity/binary ran ptxas). Keeps the contract artifact
# self-describing about which toolchain backed the shipped GPU profile.
def _tc(obj):
    t = obj.get("toolchain") if isinstance(obj, dict) else None
    return t if isinstance(t, dict) and t.get("capture_status") not in (None, "not_captured") else None
toolchain = _tc(runtime) or _tc(parity) or _tc(binary) or {"capture_status": "not_captured"}

payload = {
    "schema": "sounio.omega.gpu_public_contract.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "mode": mode,
    "status_summary": status,
    "reason": reason,
    "profile": profile,
    "checks": checks,
    "support_evidence": {
        "gpu_codegen_parity_artifact": rel(parity_path),
        "gpu_binary_attestation_artifact": rel(binary_path),
        "gpu_runtime_attestation_artifact": rel(runtime_path),
        "attested_targets": runtime.get("target_profiles", []),
        "native_lanes": runtime.get("native_lanes", parity.get("native_lanes", [])),
        "binary_provenance": binary.get("target_provenance", binary.get("binaries", [])),
        "toolchain": toolchain,
    },
    "blockers": blockers,
    "log_path": rel(log_path),
}

out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

append_check() {
  local checks_json="$1"
  local name="$2"
  local status="$3"
  local command="$4"
  local rc="$5"
  local note="$6"
  jq -cn \
    --argjson checks "$checks_json" \
    --arg name "$name" \
    --arg status "$status" \
    --arg command "$command" \
    --argjson rc "$rc" \
    --arg note "$note" \
    '$checks + [{name: $name, status: $status, command: $command, rc: $rc, note: $note}]'
}

run_ok() {
  local name="$1"
  local note="$2"
  shift 2
  local tmp_out tmp_err rc
  tmp_out="$(mktemp)"
  tmp_err="$(mktemp)"
  set +e
  "$@" >"$tmp_out" 2>"$tmp_err"
  rc=$?
  set -e
  cat "$tmp_out" >>"$LOG_PATH"
  cat "$tmp_err" >>"$LOG_PATH"
  rm -f "$tmp_out" "$tmp_err"
  printf '%s' "$rc"
}

run_expect_fail() {
  local name="$1"
  local note="$2"
  shift 2
  local tmp_out tmp_err rc
  tmp_out="$(mktemp)"
  tmp_err="$(mktemp)"
  set +e
  "$@" >"$tmp_out" 2>"$tmp_err"
  rc=$?
  set -e
  cat "$tmp_out" >>"$LOG_PATH"
  cat "$tmp_err" >>"$LOG_PATH"
  if [[ $rc -eq 0 ]]; then
    rm -f "$tmp_out" "$tmp_err"
    printf '0'
    return
  fi
  rm -f "$tmp_out" "$tmp_err"
  printf '%s' "$rc"
}

if [[ "$MODE" == "off" ]]; then
  emit_json "not_run" "gate_disabled" "[]" "[]" '{"profile":"gpu","souc_path":"","souc_version":"","public_surface":{}}'
  echo "omega_gpu_public_contract_gate: status=not_run reason=gate_disabled report=$OUT_JSON"
  exit 0
fi

GPU_SOUC=""
if GPU_SOUC="$(sounio_resolve_gpu_souc "$GPU_FIXTURE" "$GPU_SOUC_HINT" 2>/dev/null)"; then
  :
else
  probe_reason="$(sounio_gpu_probe_reason)"
  if [[ -z "$probe_reason" ]]; then
    probe_reason="gpu_backend_unavailable"
  fi
  status="not_run"
  if [[ "$MODE" == "required" ]]; then
    status="fail"
  fi
  emit_json "$status" "$probe_reason" "[\"$probe_reason\"]" "[]" '{"profile":"gpu","souc_path":"","souc_version":"","public_surface":{}}'
  if [[ "$status" == "fail" ]]; then
    echo "omega_gpu_public_contract_gate: status=fail reason=$probe_reason report=$OUT_JSON" >&2
    exit 2
  fi
  echo "omega_gpu_public_contract_gate: status=not_run reason=$probe_reason report=$OUT_JSON"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

CHECKS_JSON='[]'
BLOCKERS=()

VERSION_OUT="$("$GPU_SOUC" --version 2>>"$LOG_PATH" || true)"
INFO_OUT="$("$GPU_SOUC" info 2>>"$LOG_PATH" || true)"
BUILD_HELP_OUT="$("$GPU_SOUC" build --help 2>>"$LOG_PATH" || true)"
TOP_LEVEL_HELP_OUT="$("$GPU_SOUC" --help 2>>"$LOG_PATH" || true)"

gpu_enabled=false
jit_disabled=false
gpu_backend_public=false
gpu_emit_top_level=false

if printf '%s' "$INFO_OUT" | rg -q '\[\+\] GPU codegen' >/dev/null 2>&1; then
  gpu_enabled=true
fi
if printf '%s' "$INFO_OUT" | rg -q '\[-\] Cranelift JIT' >/dev/null 2>&1; then
  jit_disabled=true
fi
if printf '%s' "$BUILD_HELP_OUT" | rg -q -- '--backend' >/dev/null 2>&1 && printf '%s' "$BUILD_HELP_OUT" | rg -q 'gpu' >/dev/null 2>&1; then
  gpu_backend_public=true
fi
if printf '%s' "$TOP_LEVEL_HELP_OUT" | rg -q '\bgpu-emit\b' >/dev/null 2>&1; then
  gpu_emit_top_level=true
fi

CHECKS_JSON="$(append_check "$CHECKS_JSON" "info_gpu_enabled" "$([[ "$gpu_enabled" == true ]] && echo pass || echo fail)" "$GPU_SOUC info" 0 "GPU profile should report GPU codegen enabled")"
CHECKS_JSON="$(append_check "$CHECKS_JSON" "info_jit_disabled" "$([[ "$jit_disabled" == true ]] && echo pass || echo fail)" "$GPU_SOUC info" 0 "Checked GPU artifact keeps JIT disabled")"
CHECKS_JSON="$(append_check "$CHECKS_JSON" "build_help_gpu_backend" "$([[ "$gpu_backend_public" == true ]] && echo pass || echo fail)" "$GPU_SOUC build --help" 0 "Public PTX emission path is build --backend gpu")"
CHECKS_JSON="$(append_check "$CHECKS_JSON" "top_level_gpu_emit" "$([[ "$gpu_emit_top_level" == true ]] && echo pass || echo not_public)" "$GPU_SOUC --help" 0 "Top-level gpu-emit is not in the checked artifact CLI")"

if [[ "$gpu_enabled" != true || "$jit_disabled" != true || "$gpu_backend_public" != true ]]; then
  BLOCKERS+=("public_contract_mismatch")
fi

check_examples=(
  "examples/gpu.sio"
  "examples/kernel_vec_add.sio"
  "examples/kernel_matmul.sio"
  "examples/kernel_epistemic_vec_add.sio"
  "tests/run-pass/gpu_launch_surface.sio"
)

for path in "${check_examples[@]}"; do
  rc="$(run_ok "check:$path" "public_check" "$GPU_SOUC" check "$path")"
  status="pass"
  if [[ "$rc" -ne 0 ]]; then
    status="fail"
    BLOCKERS+=("public_example_fail")
  fi
  CHECKS_JSON="$(append_check "$CHECKS_JSON" "check:$path" "$status" "$GPU_SOUC check $path" "$rc" "GPU-profile source acceptance")"
done

build_examples=(
  "examples/gpu.sio"
  "examples/kernel_vec_add.sio"
  "examples/kernel_matmul.sio"
  "examples/kernel_epistemic_vec_add.sio"
)

for path in "${build_examples[@]}"; do
  base="$(basename "$path" .sio)"
  out_path="$TMP_DIR/$base.ptx"
  rc="$(run_ok "build:$path" "public_ptx_emit" "$GPU_SOUC" build "$path" --backend gpu -o "$out_path")"
  status="pass"
  note="PTX emitted via public GPU build path"
  if [[ "$rc" -ne 0 || ! -s "$out_path" ]]; then
    status="fail"
    note="PTX file missing"
    BLOCKERS+=("public_ptx_emit_fail")
  elif ! rg -q '\.entry|\.func' "$out_path" >/dev/null 2>&1; then
    status="fail"
    note="PTX markers missing"
    BLOCKERS+=("public_ptx_emit_fail")
  fi
  CHECKS_JSON="$(append_check "$CHECKS_JSON" "build:$path" "$status" "$GPU_SOUC build $path --backend gpu -o $out_path" "$rc" "$note")"
done

TMP_LAUNCH="$TMP_DIR/launch_surface.sio"
cat >"$TMP_LAUNCH" <<'EOF'
kernel fn vec_add(n: i64) with GPU {
}

fn main() with GPU, IO {
    let grid = (1, 1, 1)
    let block = (64, 1, 1)
    perform GPU.launch(vec_add, grid, block)(4)
    perform GPU.sync()
}
EOF

rc="$(run_ok "surface:launch" "gpu_launch_surface" "$GPU_SOUC" check "$TMP_LAUNCH")"
CHECKS_JSON="$(append_check "$CHECKS_JSON" "surface:GPU.launch" "$([[ "$rc" -eq 0 ]] && echo pass || echo fail)" "$GPU_SOUC check $TMP_LAUNCH" "$rc" "Checked public launch surface")"
if [[ "$rc" -ne 0 ]]; then
  BLOCKERS+=("public_contract_mismatch")
fi

TMP_INTRINSICS="$TMP_DIR/intrinsics_surface.sio"
cat >"$TMP_INTRINSICS" <<'EOF'
kernel fn vec_add(n: i64) with GPU {
    let i = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x
    if i < n {
    }
}

fn main() -> i32 with IO {
    0
}
EOF

rc="$(run_expect_fail "surface:gpu.thread_id" "gpu_intrinsics_not_public" "$GPU_SOUC" check "$TMP_INTRINSICS")"
intrinsics_status="not_public"
if [[ "$rc" -eq 0 ]]; then
  intrinsics_status="pass"
fi
CHECKS_JSON="$(append_check "$CHECKS_JSON" "surface:gpu.thread_id" "$intrinsics_status" "$GPU_SOUC check $TMP_INTRINSICS" "$rc" "Intrinsic namespace is not yet part of the checked public surface")"

TMP_ALLOC="$TMP_DIR/alloc_surface.sio"
cat >"$TMP_ALLOC" <<'EOF'
fn main() with GPU, Alloc, IO {
    let n = 4
    let a = gpu.alloc<f32>(n)
    perform GPU.sync()
}
EOF

rc="$(run_expect_fail "surface:gpu.alloc" "gpu_alloc_not_public" "$GPU_SOUC" check "$TMP_ALLOC")"
alloc_status="not_public"
if [[ "$rc" -eq 0 ]]; then
  alloc_status="pass"
fi
CHECKS_JSON="$(append_check "$CHECKS_JSON" "surface:gpu.alloc" "$alloc_status" "$GPU_SOUC check $TMP_ALLOC" "$rc" "Allocation intrinsics are not yet part of the checked public surface")"

profile_json="$(jq -cn \
  --arg souc_path "$(to_rel "$GPU_SOUC")" \
  --arg souc_version "$VERSION_OUT" \
  --arg info_output "$INFO_OUT" \
  --arg build_help_output "$BUILD_HELP_OUT" \
  --argjson gpu_enabled "$gpu_enabled" \
  --argjson jit_disabled "$jit_disabled" \
  --argjson gpu_backend_public "$gpu_backend_public" \
  --argjson gpu_emit_top_level "$gpu_emit_top_level" \
  '{
    profile: "gpu",
    souc_path: $souc_path,
    souc_version: $souc_version,
    info_excerpt: $info_output,
    build_help_excerpt: $build_help_output,
    public_surface: {
      gpu_codegen_enabled: $gpu_enabled,
      jit_disabled: $jit_disabled,
      build_backend_gpu: $gpu_backend_public,
      top_level_gpu_emit: $gpu_emit_top_level
    }
  }')"

unique_blockers='[]'
if [[ ${#BLOCKERS[@]} -gt 0 ]]; then
  unique_blockers="$(printf '%s\n' "${BLOCKERS[@]}" | sort -u | jq -R . | jq -s .)"
fi

final_status="pass"
final_reason="public_contract_verified"
if [[ $(jq -r '[.[] | select(.status == "fail")] | length' <<<"$CHECKS_JSON") -gt 0 ]]; then
  final_status="fail"
  final_reason="one_or_more_contract_checks_failed"
fi
if [[ "$MODE" == "required" && "$final_status" != "pass" ]]; then
  :
fi
if [[ "$MODE" == "auto" && "$final_status" == "fail" ]]; then
  :
fi

emit_json "$final_status" "$final_reason" "$unique_blockers" "$CHECKS_JSON" "$profile_json"

if [[ "$final_status" == "pass" ]]; then
  echo "omega_gpu_public_contract_gate: status=pass report=$OUT_JSON"
  exit 0
fi
echo "omega_gpu_public_contract_gate: status=fail reason=$final_reason report=$OUT_JSON" >&2
exit 2
