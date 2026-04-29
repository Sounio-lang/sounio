#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${SOUNIO_LOCAL_CUDA_ADMISSION_JSON:-$ROOT_DIR/artifacts/omega/local_cuda_device_admission_gate.v1.json}"
LOG_PATH="${SOUNIO_LOCAL_CUDA_ADMISSION_LOG:-$ROOT_DIR/artifacts/omega/local_cuda_device_admission_gate.log}"
RUNTIME_RUNG="${SOUNIO_LOCAL_CUDA_RUNTIME_RUNG:-epistemic_dual_output_f32_sm89_l4_flags}"
STRICT_MODE="${SOUNIO_LOCAL_CUDA_STRICT:-0}"
SKIP_SOUNIO_RUNTIME="${SOUNIO_LOCAL_CUDA_SKIP_SOUNIO_RUNTIME:-0}"
INNER_JSON="${SOUNIO_LOCAL_CUDA_SOUNIO_GATE_JSON:-$ROOT_DIR/artifacts/omega/local_cuda_device_sounio_runtime_gate.v1.json}"
INNER_LOG="${SOUNIO_LOCAL_CUDA_SOUNIO_GATE_LOG:-$ROOT_DIR/artifacts/omega/local_cuda_device_sounio_runtime_gate.log}"
INNER_CUBIN="${SOUNIO_LOCAL_CUDA_SOUNIO_CUBIN:-$ROOT_DIR/artifacts/omega/local_cuda_device_${RUNTIME_RUNG}.cubin}"
INNER_HEX="${SOUNIO_LOCAL_CUDA_SOUNIO_HEX:-$ROOT_DIR/artifacts/omega/local_cuda_device_${RUNTIME_RUNG}.hex}"
INNER_LOADER="${SOUNIO_LOCAL_CUDA_RUNTIME_LOADER:-/tmp/sounio_local_cuda_device_nvidia_bare_loader}"

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$LOG_PATH")" "$(dirname "$INNER_JSON")"
: >"$LOG_PATH"

python3 - "$ROOT_DIR" "$OUT_JSON" "$LOG_PATH" "$RUNTIME_RUNG" "$STRICT_MODE" "$SKIP_SOUNIO_RUNTIME" "$INNER_JSON" "$INNER_LOG" "$INNER_CUBIN" "$INNER_HEX" "$INNER_LOADER" <<'PY'
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
out_json = Path(sys.argv[2])
log_path = Path(sys.argv[3])
runtime_rung = sys.argv[4]
strict_mode = sys.argv[5].lower() in {"1", "true", "yes", "on"}
skip_sounio_runtime = sys.argv[6].lower() in {"1", "true", "yes", "on"}
inner_json = Path(sys.argv[7])
inner_log = Path(sys.argv[8])
inner_cubin = Path(sys.argv[9])
inner_hex = Path(sys.argv[10])
inner_loader = sys.argv[11]

def rel(path):
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)

def append_log(label, text):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {label} =====\n")
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")

def run_cmd(argv, timeout=30, env=None):
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        text = completed.stdout or ""
        append_log(" ".join(argv), text)
        return {"rc": completed.returncode, "output": text}
    except FileNotFoundError as exc:
        text = str(exc)
        append_log(" ".join(argv), text)
        return {"rc": 127, "output": text}
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "") + (exc.stderr or "")
        append_log(" ".join(argv), text)
        return {"rc": 124, "output": text, "timeout": True}

def excerpt(text, limit=2000):
    text = text.replace("\x00", "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."

def command_probe(name, version_args=("--version",)):
    path = shutil.which(name)
    result = {"present": path is not None, "path": path, "version": None, "version_rc": None}
    if path and version_args:
        run = run_cmd([name, *version_args], timeout=20)
        result["version_rc"] = run["rc"]
        result["version"] = excerpt(run["output"], 1200)
    return result

def ldconfig_paths(pattern):
    candidates = [["/sbin/ldconfig", "-p"], ["ldconfig", "-p"]]
    for argv in candidates:
        run = run_cmd(argv, timeout=20)
        if run["rc"] == 0:
            paths = []
            for line in run["output"].splitlines():
                if pattern in line:
                    m = re.search(r"=>\s+(\S+)", line)
                    if m:
                        paths.append(m.group(1))
            return sorted(set(paths))
    return []

def find_paths(*names):
    found = []
    roots = [Path("/usr/lib"), Path("/lib"), Path("/usr/local/cuda")]
    for base in roots:
        if not base.exists():
            continue
        for name in names:
            try:
                found.extend(str(p) for p in base.rglob(name))
            except PermissionError:
                pass
    return sorted(set(found))

def nvidia_pci_devices():
    devices = []
    base = Path("/sys/bus/pci/devices")
    if not base.exists():
        return devices
    for dev in sorted(base.iterdir()):
        try:
            vendor = (dev / "vendor").read_text(encoding="utf-8").strip()
            cls = (dev / "class").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if vendor == "0x10de":
            devices.append({"path": str(dev), "vendor": vendor, "class": cls})
    return devices

def device_surface():
    nodes = sorted(str(p) for p in Path("/dev").glob("nvidia*"))
    return {
        "nodes": nodes,
        "has_nvidiactl": "/dev/nvidiactl" in nodes,
        "has_nvidia0": "/dev/nvidia0" in nodes,
        "has_nvidia_uvm": "/dev/nvidia-uvm" in nodes,
        "nvidia_pci_devices": nvidia_pci_devices(),
    }

def cuinit_probe():
    result = {
        "status": "not_run",
        "reason": "libcuda_missing",
        "cuda_result": None,
        "driver_version": -1,
        "device_count": -1,
        "devices": [],
    }
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        try:
            lib = ctypes.CDLL("libcuda.so")
        except OSError as exc:
            result["load_error"] = str(exc)
            return result
    try:
        rc = int(lib.cuInit(0))
    except Exception as exc:
        result.update({"reason": "cuInit_exception", "error": str(exc)})
        return result
    result["cuda_result"] = rc
    if rc != 0:
        result["reason"] = "cuInit_failed"
        return result
    result["status"] = "pass"
    result["reason"] = "cuInit_pass"
    version = ctypes.c_int(-1)
    if int(lib.cuDriverGetVersion(ctypes.byref(version))) == 0:
        result["driver_version"] = int(version.value)
    count = ctypes.c_int(-1)
    count_rc = int(lib.cuDeviceGetCount(ctypes.byref(count)))
    if count_rc == 0:
        result["device_count"] = int(count.value)
    else:
        result["device_count_result"] = count_rc
    for ordinal in range(max(0, result["device_count"])):
        dev = ctypes.c_int()
        if int(lib.cuDeviceGet(ctypes.byref(dev), ordinal)) != 0:
            continue
        name = ctypes.create_string_buffer(128)
        major = ctypes.c_int(-1)
        minor = ctypes.c_int(-1)
        lib.cuDeviceGetName(name, 128, dev)
        lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), dev)
        result["devices"].append({
            "ordinal": ordinal,
            "name": name.value.decode("utf-8", errors="replace"),
            "cc_major": int(major.value),
            "cc_minor": int(minor.value),
        })
    return result

def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "not_run", "reason": "json_read_failed", "error": str(exc)}

def run_sounio_runtime_gate():
    if skip_sounio_runtime:
        return {
            "status": "not_run",
            "reason": "skipped_by_env",
            "json_path": rel(inner_json),
            "log_path": rel(inner_log),
        }
    env = os.environ.copy()
    env.update({
        "SOUNIO_NVIDIA_BARE_RUNTIME": "1",
        "SOUNIO_NVIDIA_BARE_RUNTIME_RUNG": runtime_rung,
        "SOUNIO_NVIDIA_BARE_GATE_JSON": str(inner_json),
        "SOUNIO_NVIDIA_BARE_GATE_LOG": str(inner_log),
        "SOUNIO_NVIDIA_BARE_CUBIN": str(inner_cubin),
        "SOUNIO_NVIDIA_BARE_HEX": str(inner_hex),
        "SOUNIO_NVIDIA_BARE_RUNTIME_LOADER": inner_loader,
    })
    if "SOUNIO_NVIDIA_BARE_VEC_N" not in env:
        env["SOUNIO_NVIDIA_BARE_VEC_N"] = os.environ.get("SOUNIO_LOCAL_CUDA_VEC_N", "64")
    run = run_cmd(["bash", "scripts/ci/native_v2_nvidia_bare_metal_gate.sh"], timeout=180, env=env)
    payload = load_json(inner_json)
    return {
        "status": payload.get("status", "not_run"),
        "reason": payload.get("reason", "unparsed"),
        "rc": run["rc"],
        "stdout_excerpt": excerpt(run["output"], 2000),
        "json_path": rel(inner_json),
        "log_path": rel(inner_log),
        "cubin_path": rel(inner_cubin),
        "hex_path": rel(inner_hex),
        "runtime": payload.get("runtime", {}),
        "structural": {
            "status": payload.get("structural", {}).get("status"),
            "reason": payload.get("structural", {}).get("reason"),
            "sha256": payload.get("structural", {}).get("sha256"),
            "size_bytes": payload.get("structural", {}).get("size_bytes"),
        },
    }

tool_versions = {
    "cc": command_probe("cc", ("--version",)),
    "nvcc": command_probe("nvcc", ("--version",)),
    "nvdisasm": command_probe("nvdisasm", ("--version",)),
    "cuobjdump": command_probe("cuobjdump", ("--version",)),
    "ptxas": command_probe("ptxas", ("--version",)),
    "nvidia-smi": command_probe("nvidia-smi", ()),
}

cuda_libraries = {
    "libcuda": {
        "ldconfig_paths": ldconfig_paths("libcuda.so"),
        "filesystem_paths": find_paths("libcuda.so", "libcuda.so.1"),
    },
    "libcudart": {
        "ldconfig_paths": ldconfig_paths("libcudart.so"),
        "filesystem_paths": find_paths("libcudart.so", "libcudart.so.13"),
    },
}
cuda_libraries["libcuda"]["present"] = bool(cuda_libraries["libcuda"]["ldconfig_paths"] or cuda_libraries["libcuda"]["filesystem_paths"])
cuda_libraries["libcudart"]["present"] = bool(cuda_libraries["libcudart"]["ldconfig_paths"] or cuda_libraries["libcudart"]["filesystem_paths"])

surface = device_surface()
nvidia_smi_list = run_cmd(["nvidia-smi", "-L"], timeout=20) if tool_versions["nvidia-smi"]["present"] else {"rc": 127, "output": "nvidia-smi missing"}
nvidia_smi_full = run_cmd(["nvidia-smi"], timeout=20) if tool_versions["nvidia-smi"]["present"] else {"rc": 127, "output": "nvidia-smi missing"}
cuinit = cuinit_probe()
sounio_gate = run_sounio_runtime_gate()

required_tools = ["cc", "nvcc", "nvdisasm", "cuobjdump", "ptxas", "nvidia-smi"]
missing_tools = [name for name in required_tools if not tool_versions[name]["present"]]
blockers = []
if missing_tools:
    blockers.append({"class": "cuda_toolchain", "reason": "missing_tools", "detail": missing_tools})
if not cuda_libraries["libcuda"]["present"]:
    blockers.append({"class": "cuda_driver_library", "reason": "libcuda_missing"})
if not surface["nodes"]:
    blockers.append({"class": "cuda_device_surface", "reason": "nvidia_device_nodes_missing"})
if not surface["nvidia_pci_devices"]:
    blockers.append({"class": "cuda_device_surface", "reason": "nvidia_pci_device_not_visible"})
if nvidia_smi_list["rc"] != 0:
    blockers.append({"class": "cuda_driver_runtime", "reason": "nvidia_smi_failed", "rc": nvidia_smi_list["rc"]})
if cuinit["status"] != "pass":
    blockers.append({"class": "cuda_driver_runtime", "reason": cuinit["reason"], "cuda_result": cuinit.get("cuda_result")})

sounio_runtime = sounio_gate.get("runtime", {})
if sounio_gate.get("status") == "fail":
    blockers.append({"class": "sounio_runtime_gate", "reason": sounio_gate.get("reason")})
elif sounio_runtime.get("status") != "pass":
    blockers.append({
        "class": "sounio_runtime_gate",
        "reason": sounio_runtime.get("reason", sounio_gate.get("reason", "not_run")),
        "cuda_result": sounio_runtime.get("cuda_result"),
    })

if sounio_runtime.get("status") == "pass":
    status = "pass"
    reason = "local_cuda_device_admitted"
elif sounio_gate.get("status") == "fail":
    status = "fail"
    reason = "sounio_runtime_gate_failed"
elif missing_tools:
    status = "not_run"
    reason = "cuda_toolchain_missing"
elif not cuda_libraries["libcuda"]["present"]:
    status = "not_run"
    reason = "cuda_driver_library_missing"
elif not surface["nodes"]:
    status = "not_run"
    reason = "nvidia_device_nodes_missing"
elif nvidia_smi_list["rc"] != 0:
    status = "not_run"
    reason = "nvidia_smi_failed"
elif cuinit["status"] != "pass":
    status = "not_run"
    reason = cuinit["reason"]
else:
    status = "not_run"
    reason = sounio_runtime.get("reason", "sounio_runtime_not_run")

payload = {
    "schema": "sounio.local-cuda-device-admission-gate.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": status,
    "reason": reason,
    "strict_mode": strict_mode,
    "proof_goal": "local CUDA Driver API device admission for Sounio-owned NVIDIA CUBIN runtime launch",
    "runtime_rung": runtime_rung,
    "host": {
        "platform": platform.platform(),
        "uname": dict(zip(["system", "node", "release", "version", "machine", "processor"], platform.uname())),
        "cwd": str(root),
    },
    "checks": {
        "toolchain": tool_versions,
        "cuda_libraries": cuda_libraries,
        "device_surface": surface,
        "nvidia_smi": {
            "list_rc": nvidia_smi_list["rc"],
            "list_output_excerpt": excerpt(nvidia_smi_list["output"], 2000),
            "full_rc": nvidia_smi_full["rc"],
            "full_output_excerpt": excerpt(nvidia_smi_full["output"], 2000),
        },
        "cuinit": cuinit,
        "sounio_runtime_gate": sounio_gate,
    },
    "artifacts": {
        "json_path": rel(out_json),
        "log_path": rel(log_path),
        "sounio_runtime_gate_json": rel(inner_json),
        "sounio_runtime_gate_log": rel(inner_log),
    },
    "blockers": blockers,
    "next_action": (
        "local CUDA launch is admitted"
        if status == "pass"
        else "expose NVIDIA host device nodes and a running NVIDIA kernel driver to this workspace, then rerun this gate"
    ),
}

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"local_cuda_device_admission_gate: status={status} reason={reason} report={out_json}")

if status == "pass":
    raise SystemExit(0)
if status == "fail" or strict_mode:
    raise SystemExit(1)
raise SystemExit(0)
PY
