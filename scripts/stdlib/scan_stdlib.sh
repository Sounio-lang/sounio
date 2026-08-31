#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

STDLIB_PATH="${ROOT_DIR}/stdlib"
JSON_OUT="${STDLIB_SCAN_JSON_OUT:-}"
QUIET=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/stdlib/scan_stdlib.sh [--stdlib-path PATH] [--json-out PATH] [--quiet]

Emits a machine-checkable snapshot of stdlib inventory:
  - total .sio files
  - total .sio.disabled files
  - stub-like mod.sio files (module-only surface)
  - active module entrypoints (lib.sio + non-stub mod.sio)
  - hypercomplex inventory (active/disabled/stub in nn/onn/qnn/snn/math lanes)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stdlib-path)
      if [[ $# -lt 2 ]]; then
        echo "error: --stdlib-path requires a value" >&2
        exit 2
      fi
      STDLIB_PATH="$2"
      shift 2
      ;;
    --json-out)
      if [[ $# -lt 2 ]]; then
        echo "error: --json-out requires a path" >&2
        exit 2
      fi
      JSON_OUT="$2"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "$STDLIB_PATH" ]]; then
  echo "error: stdlib path not found: $STDLIB_PATH" >&2
  exit 1
fi

TMP_JSON="$(mktemp)"
trap 'rm -f "$TMP_JSON"' EXIT

python3 - "$STDLIB_PATH" "$TMP_JSON" <<'PY'
import datetime
import json
from pathlib import Path
import sys

stdlib_path = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2]).resolve()

sio_files = sorted(stdlib_path.rglob("*.sio"))
disabled_files = sorted(stdlib_path.rglob("*.sio.disabled"))
mod_files = [p for p in sio_files if p.name == "mod.sio"]
lib_files = [p for p in sio_files if p.name == "lib.sio"]
entrypoint_files = sorted(mod_files + lib_files)

def is_stub_mod(path: Path) -> bool:
    code_lines = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            continue
        code_lines.append(stripped)
    return len(code_lines) == 1 and code_lines[0].startswith("module ")

stub_mod = [p for p in mod_files if is_stub_mod(p)]
stub_mod_set = {str(p) for p in stub_mod}

active_entrypoints = []
for p in entrypoint_files:
    if p.name == "lib.sio":
        active_entrypoints.append(p)
    elif str(p) not in stub_mod_set:
        active_entrypoints.append(p)

HYPER_ROOTS = {"nn", "onn", "qnn", "snn", "spnn", "quantnn", "math"}
HYPER_TOKENS = ("quaternion", "octonion", "sedenion", "hypercomplex", "g2", "clifford", "spiking", "quantum")
HYPER_LANES = ("nn", "onn", "qnn", "snn", "spnn", "quantnn", "math")

def is_hyper_path(path: Path) -> bool:
    rel = path.relative_to(stdlib_path)
    if not rel.parts:
        return False
    root = rel.parts[0]
    if root not in HYPER_ROOTS:
        return False
    if root in {"onn", "qnn", "snn", "spnn", "quantnn"}:
        return True
    joined = "/".join(rel.parts).lower()
    return any(token in joined for token in HYPER_TOKENS)

hyper_sio_files = [p for p in sio_files if is_hyper_path(p)]
hyper_disabled_files = [p for p in disabled_files if is_hyper_path(p)]
hyper_stub_mod_files = [p for p in stub_mod if is_hyper_path(p)]

def lane_for_path(path: Path) -> str:
    rel = path.relative_to(stdlib_path)
    if not rel.parts:
        return ""
    lane = rel.parts[0]
    if lane in HYPER_ROOTS:
        return lane
    return ""

lane_active = {lane: [] for lane in HYPER_LANES}
lane_disabled = {lane: [] for lane in HYPER_LANES}
lane_stub_mod = {lane: [] for lane in HYPER_LANES}

for p in hyper_sio_files:
    lane = lane_for_path(p)
    if lane in lane_active:
        lane_active[lane].append(str(p.relative_to(stdlib_path)))

for p in hyper_disabled_files:
    lane = lane_for_path(p)
    if lane in lane_disabled:
        lane_disabled[lane].append(str(p.relative_to(stdlib_path)))

for p in hyper_stub_mod_files:
    lane = lane_for_path(p)
    if lane in lane_stub_mod:
        lane_stub_mod[lane].append(str(p.relative_to(stdlib_path)))

obj = {
    "schema": "sounio.stdlib.inventory.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "command": "bash scripts/stdlib/scan_stdlib.sh",
    "stdlib_path": str(stdlib_path),
    "counts": {
        "sio_files": len(sio_files),
        "disabled_files": len(disabled_files),
        "stub_mod_files": len(stub_mod),
        "mod_files": len(mod_files),
        "lib_files": len(lib_files),
        "module_entrypoints": len(entrypoint_files),
        "active_module_entrypoints": len(active_entrypoints),
        "hyper_active_files": len(hyper_sio_files),
        "hyper_disabled_files": len(hyper_disabled_files),
        "hyper_stub_mod_files": len(hyper_stub_mod_files),
    },
    "stub_mod_paths": [str(p.relative_to(stdlib_path)) for p in stub_mod],
    "active_entrypoint_paths": [str(p.relative_to(stdlib_path)) for p in active_entrypoints],
    "hyper_active_paths": [str(p.relative_to(stdlib_path)) for p in hyper_sio_files],
    "hyper_disabled_paths": [str(p.relative_to(stdlib_path)) for p in hyper_disabled_files],
    "hyper_stub_mod_paths": [str(p.relative_to(stdlib_path)) for p in hyper_stub_mod_files],
    "hyper_lane_counts": {
        lane: {
            "active_files": len(lane_active[lane]),
            "disabled_files": len(lane_disabled[lane]),
            "stub_mod_files": len(lane_stub_mod[lane]),
        }
        for lane in HYPER_LANES
    },
    "hyper_lane_paths": {
        lane: {
            "active": lane_active[lane],
            "disabled": lane_disabled[lane],
            "stub_mod": lane_stub_mod[lane],
        }
        for lane in HYPER_LANES
    },
}
out_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY

if [[ $QUIET -eq 0 ]]; then
  python3 - "$TMP_JSON" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = obj["counts"]
print("=== STDLIB Inventory Snapshot ===")
print(f"schema: {obj['schema']}")
print(f"generated_at_utc: {obj['generated_at_utc']}")
print(f"sio_files: {counts['sio_files']}")
print(f"disabled_files: {counts['disabled_files']}")
print(f"stub_mod_files: {counts['stub_mod_files']}")
print(f"mod_files: {counts['mod_files']}")
print(f"lib_files: {counts['lib_files']}")
print(f"module_entrypoints: {counts['module_entrypoints']}")
print(f"active_module_entrypoints: {counts['active_module_entrypoints']}")
print(f"hyper_active_files: {counts['hyper_active_files']}")
print(f"hyper_disabled_files: {counts['hyper_disabled_files']}")
print(f"hyper_stub_mod_files: {counts['hyper_stub_mod_files']}")
PY
fi

if [[ -n "$JSON_OUT" ]]; then
  mkdir -p "$(dirname "$JSON_OUT")"
  cp "$TMP_JSON" "$JSON_OUT"
  if [[ $QUIET -eq 0 ]]; then
    echo "json_out: $JSON_OUT"
  fi
fi
