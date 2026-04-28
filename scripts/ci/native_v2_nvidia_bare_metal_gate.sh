#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

RUNTIME_MODE="${SOUNIO_NVIDIA_BARE_RUNTIME:-0}"
RUNTIME_RUNG="${SOUNIO_NVIDIA_BARE_RUNTIME_RUNG:-admission}"
KERNEL_KIND="${SOUNIO_NVIDIA_BARE_KERNEL_KIND:-}"
if [[ -z "$KERNEL_KIND" ]]; then
  if [[ "$RUNTIME_RUNG" == "epistemic_dual_output_f32" ]]; then
    KERNEL_KIND="epistemic_dual_output_f32"
  elif [[ "$RUNTIME_RUNG" == "vec_add_f32" || "$RUNTIME_RUNG" == "epistemic_elementwise_f32" || "$RUNTIME_RUNG" == "epistemic_dual_lane_f32" ]]; then
    KERNEL_KIND="vec_add_f32"
  else
    KERNEL_KIND="store_u32_const"
  fi
fi
case "$KERNEL_KIND" in
  store_u32_const)
    DEFAULT_STEM="sounio_bare_store_u32_const_sm80"
    KERNEL_NAME="sounio_bare_store_u32_const_sm80"
    EMITTER_HEX_LINE=1
    ;;
  vec_add_f32)
    DEFAULT_STEM="sounio_bare_vec_add_f32_sm80"
    KERNEL_NAME="sounio_bare_vec_add_f32_sm80"
    EMITTER_HEX_LINE=2
    ;;
  epistemic_dual_output_f32)
    DEFAULT_STEM="sounio_bare_epistemic_dual_output_f32_sm80"
    KERNEL_NAME="sounio_bare_epistemic_dual_output_f32_sm80"
    EMITTER_HEX_LINE=3
    ;;
  *)
    DEFAULT_STEM="sounio_bare_unknown_sm80"
    KERNEL_NAME="sounio_bare_unknown_sm80"
    EMITTER_HEX_LINE=1
    ;;
esac

OUT_JSON="${SOUNIO_NVIDIA_BARE_GATE_JSON:-$ROOT_DIR/artifacts/omega/native_v2_nvidia_bare_metal_gate.v1.json}"
OUT_CUBIN="${SOUNIO_NVIDIA_BARE_CUBIN:-$ROOT_DIR/artifacts/omega/${DEFAULT_STEM}.cubin}"
OUT_HEX="${SOUNIO_NVIDIA_BARE_HEX:-$ROOT_DIR/artifacts/omega/${DEFAULT_STEM}.hex}"
LOG_PATH="${SOUNIO_NVIDIA_BARE_GATE_LOG:-$ROOT_DIR/artifacts/omega/native_v2_nvidia_bare_metal_gate.log}"
CARTOGRAPHY_JSON="${SOUNIO_NVIDIA_BARE_CARTOGRAPHY_JSON:-$ROOT_DIR/artifacts/omega/nvidia_bare_cubin_cartography.v1.json}"
SIO_EMITTER="$ROOT_DIR/self-hosted/gpu/nvidia_bare.sio"
RUNTIME_LOADER_SRC="$ROOT_DIR/scripts/gpu/nvidia_bare_driver_loader.c"
RUNTIME_LOADER_BIN="${SOUNIO_NVIDIA_BARE_RUNTIME_LOADER:-/tmp/sounio_nvidia_bare_driver_loader}"

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
  local structural_json="$3"
  local runtime_json="$4"
  local blockers_json="$5"
  python3 - "$OUT_JSON" "$status" "$reason" "$structural_json" "$runtime_json" "$blockers_json" "$(to_rel "$OUT_CUBIN")" "$(to_rel "$OUT_HEX")" "$(to_rel "$LOG_PATH")" "$(to_rel "$CARTOGRAPHY_JSON")" "$KERNEL_NAME" "$KERNEL_KIND" <<'PY'
import json
import sys
from datetime import datetime, timezone

out, status, reason, structural, runtime, blockers, cubin, hex_path, log_path, cartography, kernel, kernel_kind = sys.argv[1:]
payload = {
    "schema": "sounio.native-v2.nvidia-bare-metal-gate.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": status,
    "reason": reason,
    "backend_identity": "nvidia_bare",
    "target": "nvidia_sm80",
    "format": "cubin_elf64",
    "kernel": kernel,
    "kernel_kind": kernel_kind,
    "proof_mode": "structural",
    "intermediate_path": "none",
    "source_emitter": "self-hosted/gpu/nvidia_bare.sio",
    "artifact_path": cubin,
    "hex_path": hex_path,
    "cartography_path": cartography,
    "structural": json.loads(structural),
    "runtime": json.loads(runtime),
    "blockers": json.loads(blockers),
    "tool_policy": {
        "forbidden_in_proof_path": ["nvcc", "ptxas", "clang", "llc", "PTX compiler APIs", ".ptx", ".metal", ".spv"],
        "inspection_only": ["nvdisasm", "cuobjdump", "readelf", "file"],
    },
    "legacy_comparison_lanes": {
        "synthetic_cubin_packers": "kept_for_comparison_not_proof",
        "ptx": "comparison_runtime_witness_not_source_of_truth",
        "apple_metal": "Apple GPU API backend, separate from omega-metal NVIDIA binary ownership",
    },
    "log_path": log_path,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

if [[ ! -x "$SOUC_BIN" ]]; then
  emit_json "not_run" "souc_unavailable" '{}' '{"status":"not_run","reason":"nvidia_driver_not_checked"}' '["souc_unavailable"]'
  echo "native_v2_nvidia_bare_metal_gate: status=not_run reason=souc_unavailable report=$OUT_JSON"
  exit 0
fi

python3 scripts/gpu/nvidia_bare_cubin_cartographer.py --root "$ROOT_DIR" --out "$(to_rel "$CARTOGRAPHY_JSON")" >>"$LOG_PATH" 2>&1

set +e
"$SOUC_BIN" check "$SIO_EMITTER" >>"$LOG_PATH" 2>&1
check_rc=$?
set -e
if [[ "$check_rc" -ne 0 ]]; then
  emit_json "fail" "sounio_emitter_check_failed" '{}' '{"status":"not_run","reason":"nvidia_driver_not_checked"}' '["sounio_emitter_check_failed"]'
  echo "native_v2_nvidia_bare_metal_gate: status=fail reason=sounio_emitter_check_failed report=$OUT_JSON"
  exit 1
fi

set +e
EMITTER_ALL_HEX="$(mktemp "${TMPDIR:-/tmp}/sounio-nvidia-bare-emit.XXXXXX.hex")"
"$SOUC_BIN" run "$SIO_EMITTER" >"$EMITTER_ALL_HEX" 2>>"$LOG_PATH"
run_rc=$?
set -e
if [[ "$run_rc" -ne 0 ]]; then
  emit_json "fail" "sounio_emitter_run_failed" '{}' '{"status":"not_run","reason":"nvidia_driver_not_checked"}' '["sounio_emitter_run_failed"]'
  echo "native_v2_nvidia_bare_metal_gate: status=fail reason=sounio_emitter_run_failed report=$OUT_JSON"
  exit 1
fi
sed -n "${EMITTER_HEX_LINE}p" "$EMITTER_ALL_HEX" >"$OUT_HEX"
if [[ ! -s "$OUT_HEX" ]]; then
  emit_json "fail" "sounio_emitter_missing_selected_artifact" '{}' '{"status":"not_run","reason":"nvidia_driver_not_checked"}' '["sounio_emitter_missing_selected_artifact"]'
  echo "native_v2_nvidia_bare_metal_gate: status=fail reason=sounio_emitter_missing_selected_artifact report=$OUT_JSON"
  exit 1
fi

scan_json="$(python3 - "$LOG_PATH" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
patterns = [
    r"\bnvcc\b", r"\bptxas\b", r"\bclang\b", r"\bllc\b",
    r"cuLink", r"nvrtc", r"\.ptx\b", r"\.metal\b", r"\.spv\b",
]
hits = [p for p in patterns if re.search(p, text)]
print(json.dumps(hits))
PY
)"
if [[ "$scan_json" != "[]" ]]; then
  emit_json "fail" "forbidden_tool_or_intermediate_in_proof_log" '{}' '{"status":"not_run","reason":"nvidia_driver_not_checked"}' "$scan_json"
  echo "native_v2_nvidia_bare_metal_gate: status=fail reason=forbidden_tool_or_intermediate_in_proof_log report=$OUT_JSON"
  exit 1
fi

structural_json="$(python3 - "$OUT_HEX" "$OUT_CUBIN" "$KERNEL_NAME" "$KERNEL_KIND" <<'PY'
import binascii
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

hex_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
kernel_name = sys.argv[3]
kernel_kind = sys.argv[4]
raw_hex = "".join(hex_path.read_text(encoding="utf-8", errors="replace").split())
if not re.fullmatch(r"[0-9a-fA-F]+", raw_hex or ""):
    print(json.dumps({"status": "fail", "reason": "non_hex_emitter_output"}))
    raise SystemExit(0)
blob = binascii.unhexlify(raw_hex)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_bytes(blob)

def cstr(table, off):
    if off >= len(table):
        return ""
    end = table.find(b"\0", off)
    if end < 0:
        end = len(table)
    return table[off:end].decode("utf-8", errors="replace")

sections = []
if len(blob) >= 64 and blob[:4] == b"\x7fELF":
    shoff = struct.unpack_from("<Q", blob, 40)[0]
    shentsize = struct.unpack_from("<H", blob, 58)[0]
    shnum = struct.unpack_from("<H", blob, 60)[0]
    shstrndx = struct.unpack_from("<H", blob, 62)[0]
    raw = []
    for i in range(shnum):
        off = shoff + i * shentsize
        if off + 64 <= len(blob):
            vals = struct.unpack_from("<IIQQQQIIQQ", blob, off)
            raw.append({"index": i, "name_offset": vals[0], "type": vals[1], "flags": vals[2], "offset": vals[4], "size": vals[5], "info": vals[7], "align": vals[8]})
    names = b""
    if 0 <= shstrndx < len(raw):
        s = raw[shstrndx]
        names = blob[s["offset"]:s["offset"] + s["size"]]
    for s in raw:
        s["name"] = cstr(names, s["name_offset"])
        sections.append(s)

markers = {
    "ptx_version": b".version" in blob,
    "ptx_target": b".target" in blob,
    "ptx_entry": b".entry" in blob,
    "metal_kernel": b"kernel void" in blob,
    "spirv_opcapability": b"OpCapability" in blob,
    "synthetic_packer": b"SOUNIO_CUDA_PACK_V1" in blob or b"SOUNIO-PTX" in blob,
}
flags = struct.unpack_from("<I", blob, 48)[0] if len(blob) >= 52 else 0
kernel_info = next((s for s in sections if s["name"] == f".nv.info.{kernel_name}"), None)
constant0 = next((s for s in sections if s["name"] == f".nv.constant0.{kernel_name}"), None)
store_info_blob = b""
if kernel_info is not None:
    store_info_blob = blob[kernel_info["offset"]:kernel_info["offset"] + kernel_info["size"]]
kind_checks = {}
if kernel_kind == "store_u32_const":
    kind_checks["store_const_42_marker_present"] = b"\x2a\x00\x00\x00" in blob
elif kernel_kind == "vec_add_f32":
    kind_checks["vec_add_f32_sass_markers_present"] = (
        b"\x81\x79\x06\x06\x04\x00\x00\x00" in blob
        and b"\x23\x72\x0b\x00\x05\x00\x00\x00" in blob
        and b"\x86\x79\x00\x08\x0b\x00\x00\x00" in blob
    )
elif kernel_kind == "epistemic_dual_output_f32":
    kind_checks["dual_output_param_size_present"] = b"\x03\x19\x40\x00" in store_info_blob
    kind_checks["dual_output_param_cbank_span_present"] = b"\x60\x01\x40\x00" in store_info_blob
    kind_checks["dual_output_eight_kparam_entries_present"] = store_info_blob.count(b"\x04\x17") == 8
    kind_checks["dual_output_two_stores_present"] = (
        (
            b"\x86\x79\x00\x0c\x0b\x00\x00\x00" in blob
            or b"\x86\x79\x00\x08\x0b\x00\x00\x00" in blob
            or b"\x86\x79\x00\x08\x0c\x00\x00\x00" in blob
        )
        and (
            b"\x86\x79\x00\x0e\x0f\x00\x00\x00" in blob
            or b"\x86\x79\x00\x0e\x07\x00\x00\x00" in blob
            or b"\x86\x79\x00\x0a\x07\x00\x00\x00" in blob
            or b"\x86\x79\x00\x0c\x07\x00\x00\x00" in blob
        )
    )
    kind_checks["dual_output_value_fadd_marker_present"] = (
        b"\x21\x72\x0b\x02\x05\x00\x00\x00" in blob
        or b"\x21\x72\x0c\x02\x05\x00\x00\x00" in blob
        or b"\x23\x72\x0b\x00\x05\x00\x00\x00" in blob
    )
    kind_checks["dual_output_uncertainty_ffma_marker_present"] = (
        b"\x23\x72\x0f\x00\x05\x00\x00\x00" in blob
        or b"\x23\x72\x07\x00\x05\x00\x00\x00" in blob
    )
else:
    kind_checks["known_kernel_kind"] = False
checks = {
    "elf_magic": blob[:4] == b"\x7fELF",
    "elf64_little_endian": len(blob) >= 6 and blob[4] == 2 and blob[5] == 1,
    "machine_cuda": len(blob) >= 20 and struct.unpack_from("<H", blob, 18)[0] == 190,
    "sm80_flags_present": (flags & 0x550) == 0x550,
    "text_section_present": any(s["name"] == f".text.{kernel_name}" for s in sections),
    "nv_info_present": any(s["name"].startswith(".nv.info") for s in sections),
    "kernel_nv_info_present": kernel_info is not None,
    "constant0_param_section_present": constant0 is not None,
    "constant0_param_window_in_bounds": constant0 is not None and constant0["size"] >= 0x174,
    "kparam_info_present": b"\x04\x17" in store_info_blob,
    "param_cbank_present": b"\x04\x0a" in store_info_blob,
    "cbank_param_size_present": b"\x03\x19" in store_info_blob,
    "no_forbidden_payload_markers": not any(markers.values()),
}
checks.update(kind_checks)
status = "pass" if all(checks.values()) else "fail"
reason = f"structural_sm80_{kernel_kind}_cubin_green" if status == "pass" else "structural_check_failed"
print(json.dumps({
    "status": status,
    "reason": reason,
    "kernel": kernel_name,
    "kernel_kind": kernel_kind,
    "sha256": hashlib.sha256(blob).hexdigest(),
    "size_bytes": len(blob),
    "elf_flags_hex": f"0x{flags:x}",
    "checks": checks,
    "forbidden_payload_markers": markers,
    "sections": sections,
}, sort_keys=True))
PY
)"

structural_status="$(python3 - "$structural_json" <<'PY'
import json, sys
print(json.loads(sys.argv[1]).get("status", "fail"))
PY
)"
structural_reason="$(python3 - "$structural_json" <<'PY'
import json, sys
print(json.loads(sys.argv[1]).get("reason", "structural_check_failed"))
PY
)"

runtime_json="$(python3 - <<'PY'
import json
import shutil
checks = {name: shutil.which(name) is not None for name in ["nvidia-smi", "nvdisasm", "cuobjdump", "nvcc", "ptxas"]}
print(json.dumps({
    "status": "not_run",
    "reason": "nvidia_driver_or_tooling_unavailable" if not checks["nvidia-smi"] else "runtime_gate_not_enabled",
    "driver_api_harness": "scripts/gpu/nvidia_bare_driver_loader.c",
    "host_tools_visible": checks,
}))
PY
)"

if [[ "$structural_status" != "pass" ]]; then
  emit_json "fail" "structural_check_failed" "$structural_json" "$runtime_json" '["structural_check_failed"]'
  echo "native_v2_nvidia_bare_metal_gate: status=fail reason=structural_check_failed report=$OUT_JSON"
  exit 1
fi

case "$RUNTIME_MODE" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    runtime_json="$(python3 - "$RUNTIME_RUNG" <<'PY'
import json
import shutil
import ctypes.util
import sys
checks = {name: shutil.which(name) is not None for name in ["nvidia-smi", "nvdisasm", "cuobjdump", "nvcc", "ptxas"]}
checks["libcuda"] = ctypes.util.find_library("cuda") is not None
print(json.dumps({
    "status": "not_run",
    "reason": "runtime_gate_disabled",
    "rung": sys.argv[1],
    "driver_api_harness": "scripts/gpu/nvidia_bare_driver_loader.c",
    "host_tools_visible": checks,
}))
PY
)"
    emit_json "pass" "${structural_reason}_runtime_not_run" "$structural_json" "$runtime_json" '[]'
    echo "native_v2_nvidia_bare_metal_gate: status=pass reason=${structural_reason}_runtime_not_run report=$OUT_JSON"
    exit 0
    ;;
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    ;;
  *)
    runtime_json="$(python3 - "$RUNTIME_MODE" <<'PY'
import json
import sys
print(json.dumps({
    "status": "not_run",
    "reason": "invalid_runtime_mode",
    "runtime_mode": sys.argv[1],
    "driver_api_harness": "scripts/gpu/nvidia_bare_driver_loader.c",
}))
PY
)"
    emit_json "fail" "invalid_runtime_mode" "$structural_json" "$runtime_json" '["invalid_runtime_mode"]'
    echo "native_v2_nvidia_bare_metal_gate: status=fail reason=invalid_runtime_mode report=$OUT_JSON"
    exit 1
    ;;
esac

case "$RUNTIME_RUNG" in
  admission|launch|store_u32_const|vec_add_f32|epistemic_elementwise_f32|epistemic_dual_lane_f32|epistemic_dual_output_f32) ;;
  *)
    runtime_json="$(python3 - "$RUNTIME_RUNG" <<'PY'
import json
import sys
print(json.dumps({
    "status": "not_run",
    "reason": "invalid_runtime_rung",
    "rung": sys.argv[1],
    "allowed_rungs": ["admission", "launch", "store_u32_const", "vec_add_f32", "epistemic_elementwise_f32", "epistemic_dual_lane_f32", "epistemic_dual_output_f32"],
    "driver_api_harness": "scripts/gpu/nvidia_bare_driver_loader.c",
}))
PY
)"
    emit_json "fail" "invalid_runtime_rung" "$structural_json" "$runtime_json" '["invalid_runtime_rung"]'
    echo "native_v2_nvidia_bare_metal_gate: status=fail reason=invalid_runtime_rung report=$OUT_JSON"
    exit 1
    ;;
esac

if command -v cc >/dev/null 2>&1; then
  set +e
  cc -O2 "$RUNTIME_LOADER_SRC" -ldl -o "$RUNTIME_LOADER_BIN" >>"$LOG_PATH" 2>&1
  cc_rc=$?
  set -e
  if [[ "$cc_rc" -ne 0 ]]; then
  runtime_json="$(python3 - "$RUNTIME_RUNG" "$cc_rc" "$(to_rel "$RUNTIME_LOADER_SRC")" "$RUNTIME_LOADER_BIN" <<'PY'
import json
import shutil
import sys
rung, rc, src, loader = sys.argv[1:]
checks = {name: shutil.which(name) is not None for name in ["nvidia-smi", "nvdisasm", "cuobjdump", "nvcc", "ptxas", "cc"]}
print(json.dumps({
    "status": "not_run",
    "reason": "runtime_loader_compile_failed",
    "rung": rung,
    "compile_rc": int(rc),
    "driver_api_harness": src,
    "loader_path": loader,
    "host_tools_visible": checks,
}))
PY
)"
  emit_json "pass" "${structural_reason}_runtime_not_run" "$structural_json" "$runtime_json" '["runtime_not_run"]'
  echo "native_v2_nvidia_bare_metal_gate: status=pass reason=${structural_reason}_runtime_not_run report=$OUT_JSON"
  exit 0
  fi
elif [[ -x "$RUNTIME_LOADER_BIN" ]]; then
  :
else
  runtime_json="$(python3 - "$RUNTIME_RUNG" <<'PY'
import json
import shutil
import sys
checks = {name: shutil.which(name) is not None for name in ["nvidia-smi", "nvdisasm", "cuobjdump", "nvcc", "ptxas", "cc"]}
print(json.dumps({
    "status": "not_run",
    "reason": "cc_missing",
    "rung": sys.argv[1],
    "driver_api_harness": "scripts/gpu/nvidia_bare_driver_loader.c",
    "host_tools_visible": checks,
}))
PY
)"
  emit_json "pass" "${structural_reason}_runtime_not_run" "$structural_json" "$runtime_json" '["runtime_not_run"]'
  echo "native_v2_nvidia_bare_metal_gate: status=pass reason=${structural_reason}_runtime_not_run report=$OUT_JSON"
  exit 0
fi

runtime_log="$(mktemp "${TMPDIR:-/tmp}/sounio-nvidia-bare-runtime.XXXXXX.log")"
set +e
"$RUNTIME_LOADER_BIN" "$OUT_CUBIN" "$KERNEL_NAME" "$RUNTIME_RUNG" >"$runtime_log" 2>&1
runtime_rc=$?
set -e
cat "$runtime_log" >>"$LOG_PATH"

runtime_json="$(python3 - "$runtime_log" "$runtime_rc" "$RUNTIME_RUNG" "$(to_rel "$RUNTIME_LOADER_SRC")" "$RUNTIME_LOADER_BIN" <<'PY'
import json
import re
import shutil
import ctypes.util
import sys
from pathlib import Path

log_path, rc, rung, src, loader = sys.argv[1:]
text = Path(log_path).read_text(encoding="utf-8", errors="replace")
status = "fail"
reason = "runtime_status_unparsed"
stage = "unknown"
cuda_result = None
match = re.search(r"sounio_nvidia_bare_runtime\s+status=(\S+)\s+reason=(\S+)\s+stage=(\S+)\s+cuda_result=(-?\d+)", text)
if match:
    status, reason, stage, cuda_result_s = match.groups()
    cuda_result = int(cuda_result_s)
elif int(rc) == 77:
    status = "not_run"
    reason = "runtime_not_run_unparsed"
elif int(rc) == 0:
    status = "pass"
    reason = "runtime_pass_unparsed"

value_match = re.search(r"observed_u32=(\d+)\s+expected_u32=(\d+)", text)
observed_u32 = int(value_match.group(1)) if value_match else None
expected_u32 = int(value_match.group(2)) if value_match else None
vec_match = re.search(r"vec_add_f32_n=(\d+)\s+vec_add_f32_max_abs_err=([0-9.eE+-]+)", text)
vec_add_f32_n = int(vec_match.group(1)) if vec_match else None
vec_add_f32_max_abs_err = float(vec_match.group(2)) if vec_match else None
epi_match = re.search(r"epistemic_elementwise_f32_n=(\d+)\s+epistemic_elementwise_f32_max_abs_err=([0-9.eE+-]+)\s+epistemic_elementwise_f32_eps_nonzero=(\d+)", text)
epistemic_elementwise_f32_n = int(epi_match.group(1)) if epi_match else None
epistemic_elementwise_f32_max_abs_err = float(epi_match.group(2)) if epi_match else None
epistemic_elementwise_f32_eps_nonzero = bool(int(epi_match.group(3))) if epi_match else None
dual_match = re.search(r"epistemic_dual_lane_f32_n=(\d+)\s+value_max_abs_err=([0-9.eE+-]+)\s+uncertainty_max_abs_err=([0-9.eE+-]+)\s+uncertainty_eps_nonzero=(\d+)", text)
epistemic_dual_lane_f32_n = int(dual_match.group(1)) if dual_match else None
epistemic_dual_lane_value_max_abs_err = float(dual_match.group(2)) if dual_match else None
epistemic_dual_lane_uncertainty_max_abs_err = float(dual_match.group(3)) if dual_match else None
epistemic_dual_lane_uncertainty_eps_nonzero = bool(int(dual_match.group(4))) if dual_match else None
dual_output_match = re.search(r"epistemic_dual_output_f32_n=(\d+)\s+value_max_abs_err=([0-9.eE+-]+)\s+uncertainty_max_abs_err=([0-9.eE+-]+)\s+uncertainty_eps_nonzero=(\d+)", text)
epistemic_dual_output_f32_n = int(dual_output_match.group(1)) if dual_output_match else None
epistemic_dual_output_value_max_abs_err = float(dual_output_match.group(2)) if dual_output_match else None
epistemic_dual_output_uncertainty_max_abs_err = float(dual_output_match.group(3)) if dual_output_match else None
epistemic_dual_output_uncertainty_eps_nonzero = bool(int(dual_output_match.group(4))) if dual_output_match else None
probe = {}
for key, conv in [
    ("driver_version", int),
    ("device_count", int),
    ("device_ordinal", int),
    ("cc_major", int),
    ("cc_minor", int),
    ("device_name", str),
]:
    m = re.search(rf"{key}=([^\s]+)", text)
    if m:
        probe[key] = conv(m.group(1))
checks = {name: shutil.which(name) is not None for name in ["nvidia-smi", "nvdisasm", "cuobjdump", "nvcc", "ptxas", "cc"]}
checks["libcuda"] = ctypes.util.find_library("cuda") is not None
print(json.dumps({
    "status": status,
    "reason": reason,
    "stage": stage,
    "cuda_result": cuda_result,
    "observed_u32": observed_u32,
    "expected_u32": expected_u32,
    "vec_add_f32_n": vec_add_f32_n,
    "vec_add_f32_max_abs_err": vec_add_f32_max_abs_err,
    "epistemic_elementwise_f32_n": epistemic_elementwise_f32_n,
    "epistemic_elementwise_f32_max_abs_err": epistemic_elementwise_f32_max_abs_err,
    "epistemic_elementwise_f32_eps_nonzero": epistemic_elementwise_f32_eps_nonzero,
    "epistemic_dual_lane_f32_n": epistemic_dual_lane_f32_n,
    "epistemic_dual_lane_value_max_abs_err": epistemic_dual_lane_value_max_abs_err,
    "epistemic_dual_lane_uncertainty_max_abs_err": epistemic_dual_lane_uncertainty_max_abs_err,
    "epistemic_dual_lane_uncertainty_eps_nonzero": epistemic_dual_lane_uncertainty_eps_nonzero,
    "epistemic_dual_output_f32_n": epistemic_dual_output_f32_n,
    "epistemic_dual_output_value_max_abs_err": epistemic_dual_output_value_max_abs_err,
    "epistemic_dual_output_uncertainty_max_abs_err": epistemic_dual_output_uncertainty_max_abs_err,
    "epistemic_dual_output_uncertainty_eps_nonzero": epistemic_dual_output_uncertainty_eps_nonzero,
    "cuda_probe": probe,
    "rung": rung,
    "rc": int(rc),
    "driver_api_harness": src,
    "loader_path": loader,
    "host_tools_visible": checks,
    "log_excerpt": text[-800:],
}))
PY
)"
runtime_status="$(python3 - "$runtime_json" <<'PY'
import json, sys
print(json.loads(sys.argv[1]).get("status", "fail"))
PY
)"
runtime_reason="$(python3 - "$runtime_json" <<'PY'
import json, sys
print(json.loads(sys.argv[1]).get("reason", "runtime_unknown"))
PY
)"

if [[ "$runtime_status" == "pass" ]]; then
  emit_json "pass" "$runtime_reason" "$structural_json" "$runtime_json" '[]'
  echo "native_v2_nvidia_bare_metal_gate: status=pass reason=$runtime_reason report=$OUT_JSON"
  exit 0
fi

if [[ "$runtime_status" == "not_run" ]]; then
  emit_json "pass" "${structural_reason}_runtime_not_run" "$structural_json" "$runtime_json" '["runtime_not_run"]'
  echo "native_v2_nvidia_bare_metal_gate: status=pass reason=${structural_reason}_runtime_not_run runtime_reason=$runtime_reason report=$OUT_JSON"
  exit 0
fi

emit_json "fail" "$runtime_reason" "$structural_json" "$runtime_json" "[\"$runtime_reason\"]"
echo "native_v2_nvidia_bare_metal_gate: status=fail reason=$runtime_reason report=$OUT_JSON"
exit 1
