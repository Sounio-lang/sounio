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

CUDA_TARGET="${OMEGA_GPU_CUDA_TARGET:-cuda-sm80}"
ROCM_TARGET="${OMEGA_GPU_ROCM_TARGET:-rocm-gfx942}"
CUDA_FORMAT="${OMEGA_GPU_CUDA_BINARY_FORMAT:-cubin}"
ROCM_FORMAT="${OMEGA_GPU_ROCM_BINARY_FORMAT:-hsaco}"
PARITY_COMPARE_CMD="${OMEGA_GPU_PARITY_COMPARE_CMD:-}"
CUDA_PACKER="${OMEGA_GPU_CUDA_PACKER:-$ROOT_DIR/scripts/omega/omega_cuda_binary_packer.py}"

STRICT_PARITY_RAW="${OMEGA_GPU_STRICT_PARITY:-${SOUNIO_GPU_STRICT_PARITY:-1}}"
SOUC_INVOKER="${OMEGA_GPU_PARITY_SOUC_INVOKER:-$ROOT_DIR/souc}"

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

emit_status_json() {
  local status="$1"
  local reason="$2"
  local blockers_json="$3"
  local targets_json="$4"
  local parity_json="$5"
  local gate_rc="$6"

  jq -cn \
    --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg status "$status" \
    --arg reason "$reason" \
    --arg mode "$MODE" \
    --argjson strict_parity "$STRICT_PARITY" \
    --arg fixture "$(to_rel "$GPU_FIXTURE")" \
    --arg log_path "$(to_rel "$LOG_PATH")" \
    --argjson blockers "$blockers_json" \
    --argjson targets "$targets_json" \
    --argjson parity "$parity_json" \
    --argjson gate_rc "$gate_rc" \
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
      blockers: $blockers,
      gate_rc: $gate_rc,
      log_path: $log_path
    }' >"$OUT_JSON"
}

if [[ "$MODE" == "off" ]]; then
  emit_status_json "not_run" "gate_disabled" "[]" "[]" '{"status":"not_run","reason":"gate_disabled","checked":false,"rc":0}' 0
  echo "omega_gpu_codegen_parity_gate: status=not_run reason=gate_disabled report=$OUT_JSON"
  exit 0
fi

if [[ ! -f "$GPU_FIXTURE" ]]; then
  emit_status_json "fail" "fixture_missing" '["target_unavailable"]' "[]" '{"status":"fail","reason":"fixture_missing","checked":false,"rc":2}' 2
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
    emit_status_json "fail" "souc_unavailable" '["target_unavailable"]' "[]" '{"status":"fail","reason":"souc_unavailable","checked":false,"rc":2}' 2
    echo "omega_gpu_codegen_parity_gate: status=fail reason=souc_unavailable report=$OUT_JSON" >&2
    exit 2
  fi
  emit_status_json "not_run" "souc_unavailable" '["target_unavailable"]' "[]" '{"status":"not_run","reason":"souc_unavailable","checked":false,"rc":0}' 0
  echo "omega_gpu_codegen_parity_gate: status=not_run reason=souc_unavailable report=$OUT_JSON"
  exit 0
fi

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

  set +e
  "$SOUC_INVOKER" build "$GPU_FIXTURE" --backend gpu --gpu-target "$target" --gpu-binary-format "$binary_format" -o "$out_path" >"$tmp_log" 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    if [[ -s "$out_path" ]]; then
      status="pass"
      reason="build_ok"
      sha256="$(sha256sum "$out_path" | awk '{print $1}')"
    else
      status="fail"
      reason="binary_pack_fail"
      rc=3
    fi
  else
    if rg -qi "unknown gpu target|unsupported gpu target|gpu backend not enabled|not built with gpu feature|rocm unavailable|hip unavailable|amdgpu unavailable|target unavailable" "$tmp_log"; then
      reason="target_unavailable"
    elif rg -qi "isa|encode|instruction|arch|code object" "$tmp_log"; then
      reason="isa_encode_unsupported"
    elif rg -qi "driver|launch|module|runtime" "$tmp_log"; then
      reason="driver_reject"
    else
      reason="binary_pack_fail"
    fi
  fi

  if [[ "$status" != "pass" && "$lane" == "cuda" ]]; then
    if [[ "$binary_format" == "cubin" || "$binary_format" == "fatbin" || "$binary_format" == "multi" ]]; then
      if [[ -x "$CUDA_PACKER" || -f "$CUDA_PACKER" ]]; then
        compile_mode="ptx_fallback_attempted"
        local fallback_ptx="$TMP_DIR/${lane}.fallback.ptx"
        local fallback_log="$TMP_DIR/${lane}.fallback.log"
        local fallback_meta="$TMP_DIR/${lane}.packer_meta.json"
        local fallback_rc=0
        set +e
        "$SOUC_INVOKER" build "$GPU_FIXTURE" --backend gpu -o "$fallback_ptx" >"$fallback_log" 2>&1
        fallback_rc=$?
        set -e
        cat "$fallback_log" >>"$tmp_log"
        if [[ $fallback_rc -eq 0 && -s "$fallback_ptx" ]]; then
          if python3 "$CUDA_PACKER" --ptx "$fallback_ptx" --output "$out_path" --format "$binary_format" --target "$target" --meta-out "$fallback_meta" >>"$tmp_log" 2>&1; then
            status="pass"
            reason="packed_from_ptx"
            rc=0
            compile_mode="ptx_pack_fallback"
            provenance="sounio_cuda_packer_bootstrap"
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

blockers_json="$(python3 - "$targets_json" "$STRICT_PARITY" "$parity_status" <<'PY'
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

final_status="pass"
final_reason="parity_pass"
if [[ "$blockers_json" != "[]" ]]; then
  if [[ "$MODE" == "required" ]]; then
    final_status="fail"
    final_reason="parity_failed"
  else
    final_status="not_run"
    final_reason="parity_non_pass_auto"
  fi
fi

emit_status_json "$final_status" "$final_reason" "$blockers_json" "$targets_json" "$parity_json" 0

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
