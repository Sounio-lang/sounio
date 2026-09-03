#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: required dependency missing: jq" >&2
  exit 2
fi

MODE="${OMEGA_GPU_CODEGEN_PARITY_MODE:-auto}"
OUT_JSON="${OMEGA_GPU_CODEGEN_PARITY_OUT:-$ROOT_DIR/artifacts/omega/gpu_codegen_parity.v1.json}"
LOG_PATH="${OMEGA_GPU_CODEGEN_PARITY_LOG:-$ROOT_DIR/artifacts/omega/gpu_codegen_parity_gate.log}"
GPU_FIXTURE="${OMEGA_GPU_FIXTURE:-$ROOT_DIR/scripts/fixtures/gpu_minimal.sio}"
BIN_DIR="${OMEGA_GPU_CODEGEN_BIN_DIR:-$ROOT_DIR/artifacts/omega/gpu_bins}"
NATIVE_LANE_MATRIX_JSON="${OMEGA_NATIVE_LANE_MATRIX_JSON:-$ROOT_DIR/artifacts/stdlib/native_lane_matrix.v1.json}"
REQUIRED_NATIVE_LANES="${OMEGA_REQUIRED_NATIVE_LANES:-onn,qnn,snn,spnn,quantnn,hyper_math,exceptional}"

CUDA_TARGET="${OMEGA_GPU_CUDA_TARGET:-cuda-sm80}"
ROCM_TARGET="${OMEGA_GPU_ROCM_TARGET:-rocm-gfx942}"
CUDA_FORMAT="${OMEGA_GPU_CUDA_BINARY_FORMAT:-cubin}"
ROCM_FORMAT="${OMEGA_GPU_ROCM_BINARY_FORMAT:-hsaco}"
PARITY_COMPARE_CMD="${OMEGA_GPU_PARITY_COMPARE_CMD:-}"
CUDA_PACKER="${OMEGA_GPU_CUDA_PACKER:-$ROOT_DIR/scripts/omega/omega_cuda_binary_packer.py}"
ROCM_PACKER="${OMEGA_GPU_ROCM_PACKER:-$ROOT_DIR/scripts/omega/omega_rocm_binary_packer.py}"

STRICT_PARITY_RAW="${OMEGA_GPU_STRICT_PARITY:-${SOUNIO_GPU_STRICT_PARITY:-1}}"
SOUC_INVOKER="${OMEGA_GPU_PARITY_SOUC_INVOKER:-$ROOT_DIR/souc}"
GPU_SOUC_HINT="${OMEGA_GPU_PARITY_SOUC_BIN:-}"

normalize_bool() {
  local raw="$1"
  case "$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      printf '1'
      ;;
    0|false|no|off)
      printf '0'
      ;;
    *)
      return 1
      ;;
  esac
}

STRICT_PARITY="$(normalize_bool "$STRICT_PARITY_RAW" || true)"
if [[ -z "$STRICT_PARITY" ]]; then
  echo "error: invalid OMEGA_GPU_STRICT_PARITY=$STRICT_PARITY_RAW (expected boolean)" >&2
  exit 2
fi

case "$MODE" in
  auto|required|off) ;;
  *)
    echo "error: OMEGA_GPU_CODEGEN_PARITY_MODE must be auto|required|off (got '$MODE')" >&2
    exit 2
    ;;
esac

case "$CUDA_FORMAT" in
  fatbin|cubin|hsaco|multi) ;;
  *)
    echo "error: OMEGA_GPU_CUDA_BINARY_FORMAT must be fatbin|cubin|hsaco|multi" >&2
    exit 2
    ;;
esac
case "$ROCM_FORMAT" in
  fatbin|cubin|hsaco|multi) ;;
  *)
    echo "error: OMEGA_GPU_ROCM_BINARY_FORMAT must be fatbin|cubin|hsaco|multi" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$LOG_PATH")" "$BIN_DIR"
: >"$LOG_PATH"

to_rel() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${path#$ROOT_DIR/}"
  else
    printf '%s' "$path"
  fi
}

log_matches() {
  local pattern="$1"
  local file_path="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -qi "$pattern" "$file_path"
  else
    grep -Eqi "$pattern" "$file_path"
  fi
}

detect_output_kind() {
  local file_path="$1"
  python3 - "$file_path" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print("missing")
    raise SystemExit(0)
raw = path.read_bytes()
if len(raw) == 0:
    print("empty")
    raise SystemExit(0)
if raw.startswith(b"\x7fELF"):
    print("elf")
    raise SystemExit(0)
head = raw[:256]
try:
    text = head.decode("utf-8", errors="ignore")
except Exception:
    text = ""
if ".version" in text and ".target" in text:
    print("ptx")
elif text.strip():
    print("text")
else:
    print("binary")
PY
}

validate_output_kind() {
  local lane="$1"
  local binary_format="$2"
  local output_kind="$3"

  case "$binary_format" in
    hsaco|cubin)
      [[ "$output_kind" == "elf" ]]
      return
      ;;
    fatbin|multi)
      [[ "$output_kind" == "elf" || "$output_kind" == "binary" ]]
      return
      ;;
    *)
      [[ "$output_kind" != "missing" && "$output_kind" != "empty" ]]
      return
      ;;
  esac
}

emit_status_json() {
  local status="$1"
  local reason="$2"
  local blockers_json="$3"
  local targets_json="$4"
  local parity_json="$5"
  local native_lanes_json="$6"
  local gate_rc="$7"

  jq -cn \
    --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg status "$status" \
    --arg reason "$reason" \
    --arg mode "$MODE" \
    --argjson strict_parity "$STRICT_PARITY" \
    --arg fixture "$(to_rel "$GPU_FIXTURE")" \
    --arg native_lane_matrix_artifact "$(to_rel "$NATIVE_LANE_MATRIX_JSON")" \
    --arg log_path "$(to_rel "$LOG_PATH")" \
    --argjson blockers "$blockers_json" \
    --argjson targets "$targets_json" \
    --argjson parity "$parity_json" \
    --argjson native_lanes "$native_lanes_json" \
    --argjson gate_rc "$gate_rc" \
    --argjson toolchain "$TOOLCHAIN_JSON" \
    '{
      schema: "sounio.omega.gpu_codegen_parity.v1",
      generated_at_utc: $generated_at_utc,
      mode: $mode,
      status_summary: $status,
      reason: $reason,
      strict_parity: ($strict_parity == 1),
      fixture: $fixture,
      targets: $targets,
      parity: $parity,
      native_lane_matrix_artifact: $native_lane_matrix_artifact,
      native_lanes: $native_lanes,
      toolchain: $toolchain,
      blockers: $blockers,
      gate_rc: $gate_rc,
      log_path: $log_path
    }' >"$OUT_JSON"
}

# CUDA toolchain provenance of the host that ran ptxas to produce these PTX/CUBIN
# outputs (output_sha256 is a function of the ptxas version → only comparable across
# toolkit upgrades with the toolchain recorded). Env override → local capture → not_captured.
TOOLCHAIN_JSON="${OMEGA_GPU_TOOLCHAIN_JSON:-}"
[[ -z "$TOOLCHAIN_JSON" ]] && TOOLCHAIN_JSON="$("$ROOT_DIR/scripts/omega/omega_capture_toolchain.sh" 2>/dev/null || true)"
[[ -z "$TOOLCHAIN_JSON" ]] && TOOLCHAIN_JSON='{"capture_status":"not_captured"}'

if [[ "$MODE" == "off" ]]; then
  emit_status_json "not_run" "gate_disabled" "[]" "[]" '{"status":"not_run","reason":"gate_disabled","checked":false,"rc":0}' "[]" 0
  echo "omega_gpu_codegen_parity_gate: status=not_run reason=gate_disabled report=$OUT_JSON"
  exit 0
fi

if [[ ! -f "$GPU_FIXTURE" ]]; then
  emit_status_json "fail" "fixture_missing" '["target_unavailable"]' "[]" '{"status":"fail","reason":"fixture_missing","checked":false,"rc":2}' "[]" 2
  echo "error: missing GPU fixture: $GPU_FIXTURE" >&2
  exit 2
fi

if [[ ! -x "$SOUC_INVOKER" ]]; then
  if [[ -x "${SOUC_BIN:-}" ]]; then
    SOUC_INVOKER="$SOUC_BIN"
  fi
fi
if [[ ! -x "$SOUC_INVOKER" ]]; then
  if [[ "$MODE" == "required" ]]; then
    emit_status_json "fail" "souc_unavailable" '["target_unavailable"]' "[]" '{"status":"fail","reason":"souc_unavailable","checked":false,"rc":2}' "[]" 2
    echo "omega_gpu_codegen_parity_gate: status=fail reason=souc_unavailable report=$OUT_JSON" >&2
    exit 2
  fi
  emit_status_json "not_run" "souc_unavailable" '["target_unavailable"]' "[]" '{"status":"not_run","reason":"souc_unavailable","checked":false,"rc":0}' "[]" 0
  echo "omega_gpu_codegen_parity_gate: status=not_run reason=souc_unavailable report=$OUT_JSON"
  exit 0
fi

GPU_SOUC_CANDIDATE="$SOUC_INVOKER"
if [[ -n "$GPU_SOUC_HINT" ]]; then
  GPU_SOUC_CANDIDATE="$GPU_SOUC_HINT"
fi

if ! SOUC_GPU_RESOLVED="$(sounio_resolve_gpu_souc "$GPU_FIXTURE" "$GPU_SOUC_CANDIDATE" 2>/dev/null)"; then
  gpu_probe_reason="$(sounio_gpu_probe_reason)"
  [[ -z "$gpu_probe_reason" ]] && gpu_probe_reason="gpu_backend_unavailable"
  if [[ "$MODE" == "required" ]]; then
    emit_status_json "fail" "$gpu_probe_reason" '["gpu_backend_unavailable"]' "[]" '{"status":"fail","reason":"gpu_backend_unavailable","checked":false,"rc":2}' "[]" 2
    echo "omega_gpu_codegen_parity_gate: status=fail reason=$gpu_probe_reason report=$OUT_JSON" >&2
    exit 2
  fi
  emit_status_json "not_run" "$gpu_probe_reason" '["gpu_backend_unavailable"]' "[]" '{"status":"not_run","reason":"gpu_backend_unavailable","checked":false,"rc":0}' "[]" 0
  echo "omega_gpu_codegen_parity_gate: status=not_run reason=$gpu_probe_reason report=$OUT_JSON"
  exit 0
fi
SOUC_INVOKER="$SOUC_GPU_RESOLVED"

BUILD_HELP_TEXT="$("$SOUC_INVOKER" build --help 2>/dev/null || true)"
SOUC_HAS_GPU_TARGET_FLAG=0
SOUC_HAS_GPU_BINARY_FORMAT_FLAG=0
SOUC_HAS_TARGET_FLAG=0
[[ "$BUILD_HELP_TEXT" == *"--gpu-target"* ]] && SOUC_HAS_GPU_TARGET_FLAG=1
[[ "$BUILD_HELP_TEXT" == *"--gpu-binary-format"* ]] && SOUC_HAS_GPU_BINARY_FORMAT_FLAG=1
[[ "$BUILD_HELP_TEXT" == *"--target"* ]] && SOUC_HAS_TARGET_FLAG=1

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

declare -a TARGET_ROWS

timestamp_utc="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

run_target_compile() {
  local lane="$1"
  local target="$2"
  local binary_format="$3"

  local out_path="$BIN_DIR/${lane}.${timestamp_utc}.${binary_format}"
  local tmp_log="$TMP_DIR/${lane}.log"
  local gate_log="$ROOT_DIR/artifacts/omega/gpu_codegen_parity_${lane}.log"
  local status="fail"
  local reason="binary_pack_fail"
  local rc=0
  local sha256=""
  local compile_mode="direct_build"
  local provenance="souc_gpu_build"
  local packer_schema=""
  local ptx_fallback_path=""
  local ptx_fallback_sha256=""
  local output_kind="missing"
  local cli_mode="legacy_default"
  local -a build_cmd
  build_cmd=("$SOUC_INVOKER" build "$GPU_FIXTURE" --backend gpu)
  if [[ "$SOUC_HAS_GPU_TARGET_FLAG" -eq 1 ]]; then
    build_cmd+=(--gpu-target "$target")
    cli_mode="gpu_target"
  elif [[ "$SOUC_HAS_TARGET_FLAG" -eq 1 ]]; then
    build_cmd+=(--target "$target")
    cli_mode="target"
  fi
  if [[ "$SOUC_HAS_GPU_BINARY_FORMAT_FLAG" -eq 1 ]]; then
    build_cmd+=(--gpu-binary-format "$binary_format")
    cli_mode="${cli_mode}_format"
  fi
  build_cmd+=(-o "$out_path")
  compile_mode="direct_build_${cli_mode}"

  set +e
  "${build_cmd[@]}" >"$tmp_log" 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    if [[ -s "$out_path" ]]; then
      output_kind="$(detect_output_kind "$out_path")"
      if validate_output_kind "$lane" "$binary_format" "$output_kind"; then
        status="pass"
        reason="build_ok"
        sha256="$(sha256sum "$out_path" | awk '{print $1}')"
      else
        status="fail"
        rc=3
        if [[ "$lane" == "rocm" ]]; then
          reason="isa_encode_unsupported"
        else
          reason="binary_pack_fail"
        fi
      fi
    else
      status="fail"
      reason="binary_pack_fail"
      rc=3
    fi
  else
    if log_matches "gpu backend not enabled|not built with gpu feature|not built with gpu support" "$tmp_log"; then
      reason="gpu_backend_unavailable"
    elif log_matches "unknown gpu target|unsupported gpu target|rocm unavailable|hip unavailable|amdgpu unavailable|target unavailable|unexpected argument '--gpu-target'|unexpected argument '--gpu-binary-format'" "$tmp_log"; then
      reason="target_unavailable"
    elif log_matches "isa|encode|instruction|arch|code object" "$tmp_log"; then
      reason="isa_encode_unsupported"
    elif log_matches "driver|launch|module|runtime" "$tmp_log"; then
      reason="driver_reject"
    else
      reason="binary_pack_fail"
    fi
  fi

  if [[ "$status" != "pass" ]]; then
    local selected_packer=""
    local selected_provenance=""
    local can_fallback=0
    if [[ "$lane" == "cuda" ]]; then
      if [[ "$binary_format" == "cubin" || "$binary_format" == "fatbin" || "$binary_format" == "multi" ]]; then
        if [[ -x "$CUDA_PACKER" || -f "$CUDA_PACKER" ]]; then
          selected_packer="$CUDA_PACKER"
          selected_provenance="sounio_cuda_packer_bootstrap"
          can_fallback=1
        fi
      fi
    elif [[ "$lane" == "rocm" ]]; then
      if [[ "$binary_format" == "hsaco" || "$binary_format" == "multi" ]]; then
        if [[ -x "$ROCM_PACKER" || -f "$ROCM_PACKER" ]]; then
          selected_packer="$ROCM_PACKER"
          selected_provenance="sounio_rocm_packer_bootstrap"
          can_fallback=1
        fi
      fi
    fi

    if [[ "$can_fallback" -eq 1 ]]; then
      compile_mode="ptx_fallback_attempted"
      local fallback_ptx="$TMP_DIR/${lane}.fallback.ptx"
      local fallback_log="$TMP_DIR/${lane}.fallback.log"
      local fallback_meta="$TMP_DIR/${lane}.packer_meta.json"
      local fallback_rc=0
      local -a fallback_build_cmd
      fallback_build_cmd=("$SOUC_INVOKER" build "$GPU_FIXTURE" --backend gpu)
      if [[ "$SOUC_HAS_GPU_TARGET_FLAG" -eq 1 ]]; then
        fallback_build_cmd+=(--gpu-target "$target")
      elif [[ "$SOUC_HAS_TARGET_FLAG" -eq 1 ]]; then
        fallback_build_cmd+=(--target "$target")
      fi
      fallback_build_cmd+=(-o "$fallback_ptx")

      set +e
      "${fallback_build_cmd[@]}" >"$fallback_log" 2>&1
      fallback_rc=$?
      set -e
      cat "$fallback_log" >>"$tmp_log"
      if [[ $fallback_rc -eq 0 && -s "$fallback_ptx" ]]; then
        if python3 "$selected_packer" --ptx "$fallback_ptx" --output "$out_path" --format "$binary_format" --target "$target" --meta-out "$fallback_meta" >>"$tmp_log" 2>&1; then
          status="pass"
          reason="packed_from_ptx"
          rc=0
          compile_mode="ptx_pack_fallback"
          provenance="$selected_provenance"
          output_kind="$(detect_output_kind "$out_path")"
          sha256="$(sha256sum "$out_path" | awk '{print $1}')"
          ptx_fallback_path="$(to_rel "$fallback_ptx")"
          ptx_fallback_sha256="$(sha256sum "$fallback_ptx" | awk '{print $1}')"
          if [[ -f "$fallback_meta" ]]; then
            packer_schema="$(jq -r '.schema // ""' "$fallback_meta" 2>/dev/null || true)"
          fi
        else
          compile_mode="ptx_pack_fallback_failed"
          reason="binary_pack_fail"
          rc=4
        fi
      else
        compile_mode="ptx_fallback_unavailable"
      fi
    fi
  fi

  mkdir -p "$(dirname "$gate_log")"
  cp "$tmp_log" "$gate_log"

  local target_json
  target_json="$(jq -cn \
    --arg lane "$lane" \
    --arg target "$target" \
    --arg binary_format "$binary_format" \
    --arg status "$status" \
    --arg reason "$reason" \
    --arg output_path "$(to_rel "$out_path")" \
    --arg output_sha256 "$sha256" \
    --arg log_path "$(to_rel "$gate_log")" \
    --arg compile_mode "$compile_mode" \
    --arg provenance "$provenance" \
    --arg packer_schema "$packer_schema" \
    --arg ptx_fallback_path "$ptx_fallback_path" \
    --arg ptx_fallback_sha256 "$ptx_fallback_sha256" \
    --arg output_kind "$output_kind" \
    --argjson rc "$rc" \
    '{
      lane: $lane,
      target: $target,
      binary_format: $binary_format,
      status: $status,
      reason: $reason,
      rc: $rc,
      output_path: $output_path,
      output_sha256: $output_sha256,
      compile_mode: $compile_mode,
      provenance: $provenance,
      packer_schema: $packer_schema,
      ptx_fallback_path: $ptx_fallback_path,
      ptx_fallback_sha256: $ptx_fallback_sha256,
      output_kind: $output_kind,
      log_path: $log_path
    }')"

  TARGET_ROWS+=("$target_json")
  cat "$tmp_log" >>"$LOG_PATH"
  printf '\n' >>"$LOG_PATH"
}

run_target_compile "cuda" "$CUDA_TARGET" "$CUDA_FORMAT"
run_target_compile "rocm" "$ROCM_TARGET" "$ROCM_FORMAT"

joined_targets=""
for row in "${TARGET_ROWS[@]}"; do
  if [[ -n "$joined_targets" ]]; then
    joined_targets+=","
  fi
  joined_targets+="$row"
done
targets_json="[$joined_targets]"

cuda_status="$(jq -r '.[] | select(.lane=="cuda") | .status' <<<"$targets_json")"
rocm_status="$(jq -r '.[] | select(.lane=="rocm") | .status' <<<"$targets_json")"

parity_checked=false
parity_status="not_run"
parity_reason="insufficient_targets"
parity_rc=0

if [[ "$cuda_status" == "pass" && "$rocm_status" == "pass" ]]; then
  parity_checked=true
  parity_status="pass"
  parity_reason="build_outputs_present"
  if [[ -n "$PARITY_COMPARE_CMD" ]]; then
    set +e
    CUDA_OUTPUT_PATH="$(jq -r '.[] | select(.lane=="cuda") | .output_path' <<<"$targets_json")" \
    ROCM_OUTPUT_PATH="$(jq -r '.[] | select(.lane=="rocm") | .output_path' <<<"$targets_json")" \
      bash -lc "$PARITY_COMPARE_CMD" >>"$LOG_PATH" 2>&1
    parity_rc=$?
    set -e
    if [[ $parity_rc -ne 0 ]]; then
      parity_status="fail"
      parity_reason="parity_fail"
    else
      parity_reason="custom_compare_pass"
    fi
  fi
fi

parity_json="$(jq -cn --arg status "$parity_status" --arg reason "$parity_reason" --argjson checked "$parity_checked" --argjson rc "$parity_rc" '{status:$status,reason:$reason,checked:$checked,rc:$rc}')"

target_blockers_json="$(python3 - "$targets_json" "$STRICT_PARITY" "$parity_status" <<'PY'
import json
import sys

targets = json.loads(sys.argv[1])
strict = sys.argv[2] == "1"
parity_status = sys.argv[3]
blockers = []
for row in targets:
    if row.get("status") != "pass":
        reason = row.get("reason") or "binary_pack_fail"
        if reason not in blockers:
            blockers.append(reason)
if strict and parity_status != "pass":
    if "parity_fail" not in blockers:
        blockers.append("parity_fail")
print(json.dumps(blockers, separators=(",", ":")))
PY
)"

native_lane_analysis_json="$(python3 - "$NATIVE_LANE_MATRIX_JSON" "$REQUIRED_NATIVE_LANES" "$ROOT_DIR" "$SOUC_INVOKER" <<'PY'
import json
import subprocess
from pathlib import Path
import sys

matrix_path = Path(sys.argv[1])
required_lanes = [lane.strip() for lane in sys.argv[2].split(",") if lane.strip()]
root = Path(sys.argv[3]).resolve()
souc = Path(sys.argv[4]).resolve()
alias = {"hyper_math": "math"}

lanes_out = []
blockers = []

def blocker_for(status: str, reason: str) -> str:
    text = (reason or "").lower()
    if status == "not_run" or "missing" in text or "not_run" in text:
        return "native_lane_missing"
    if "compile" in text or "check_failed" in text:
        return "native_lane_compile_fail"
    if "parity" in text or "golden" in text:
        return "native_lane_parity_fail"
    if "perf" in text or "throughput" in text:
        return "native_lane_perf_regression"
    return "native_lane_runtime_fail"

rows = {}
if matrix_path.exists():
    try:
        obj = json.loads(matrix_path.read_text(encoding="utf-8", errors="replace"))
        for row in obj.get("lanes", []) if isinstance(obj.get("lanes"), list) else []:
            if isinstance(row, dict):
                lane = str(row.get("lane", ""))
                if lane:
                    rows[lane] = row
    except Exception:
        rows = {}

for lane in required_lanes:
    lookup = alias.get(lane, lane)
    row = rows.get(lookup)
    if row is None and lane == "exceptional":
        test_path = root / "self-hosted/native/test_exceptional.sio"
        bench_path = root / "self-hosted/native/bench_exceptional.sio"
        if souc.exists() and test_path.exists() and bench_path.exists():
            test_rc = subprocess.run(
                [str(souc), "check", str(test_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            bench_rc = subprocess.run(
                [str(souc), "check", str(bench_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            if test_rc == 0 and bench_rc == 0:
                row = {
                    "lane": "exceptional",
                    "status": "pass",
                    "reason": "compile_ok",
                    "required": True,
                }
            else:
                row = {
                    "lane": "exceptional",
                    "status": "fail",
                    "reason": "native_lane_compile_fail",
                    "required": True,
                    "blocker": "native_lane_compile_fail",
                }

    status = "not_run"
    reason = "native_lane_missing"
    blocker = "native_lane_missing"
    if row is not None:
        status = str(row.get("status", "not_run"))
        reason = str(row.get("reason", "native_lane_runtime_fail"))
        blocker = blocker_for(status, reason)
    if status != "pass" and blocker not in blockers:
        blockers.append(blocker)
    lanes_out.append({
        "lane": lane,
        "required": True,
        "status": status,
        "reason": reason,
        "blocker": blocker if status != "pass" else "",
        "matrix_lane": lookup,
    })

status = "pass" if not blockers else "fail"
reason = "native_lanes_pass" if not blockers else blockers[0]
print(json.dumps({
    "lanes": lanes_out,
    "blockers": blockers,
    "status": status,
    "reason": reason,
}, separators=(",", ":")))
PY
)"

native_lanes_json="$(python3 - "$native_lane_analysis_json" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(json.dumps(obj.get("lanes", []), separators=(",", ":")))
PY
)"

native_lane_blockers_json="$(python3 - "$native_lane_analysis_json" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(json.dumps(obj.get("blockers", []), separators=(",", ":")))
PY
)"

blockers_json="$(python3 - "$target_blockers_json" "$native_lane_blockers_json" <<'PY'
import json
import sys

merged = []
for raw in (sys.argv[1], sys.argv[2]):
    arr = json.loads(raw)
    for item in arr:
        if item not in merged:
            merged.append(item)
print(json.dumps(merged, separators=(",", ":")))
PY
)"

first_blocker="$(python3 - "$blockers_json" <<'PY'
import json
import sys
arr = json.loads(sys.argv[1])
print(arr[0] if arr else "")
PY
)"

final_status="pass"
final_reason="parity_pass"
if [[ "$blockers_json" != "[]" ]]; then
  if [[ "$MODE" == "required" ]]; then
    final_status="fail"
    final_reason="${first_blocker:-parity_failed}"
  else
    final_status="not_run"
    final_reason="parity_non_pass_auto"
  fi
fi

emit_status_json "$final_status" "$final_reason" "$blockers_json" "$targets_json" "$parity_json" "$native_lanes_json" 0

if [[ "$final_status" == "pass" ]]; then
  echo "omega_gpu_codegen_parity_gate: status=pass report=$OUT_JSON"
  exit 0
fi
if [[ "$final_status" == "not_run" ]]; then
  echo "omega_gpu_codegen_parity_gate: status=not_run reason=$final_reason report=$OUT_JSON"
  exit 0
fi

echo "omega_gpu_codegen_parity_gate: status=fail reason=$final_reason report=$OUT_JSON" >&2
exit 2
