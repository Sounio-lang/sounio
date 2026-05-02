#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-imported-core-abi] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-imported-core-abi] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

OUT_DIR="${SOUNIO_NATIVE_V2_IMPORTED_CORE_ABI_DIR:-$(mktemp -d /tmp/sounio-native-v2-imported-core-abi.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
PROGRAM="tests/selfhost/native_runtime/import_core_abi_42.sio"
BUNDLE="$ARTIFACT_DIR/import_core_abi_42.bundle.sio"
ELF="$ARTIFACT_DIR/import_core_abi_42.native"
SUMMARY_JSON="$ARTIFACT_DIR/native_v2_imported_core_abi.v1.json"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

echo "[native-v2-imported-core-abi] souc=$SOUC_BIN"
echo "[native-v2-imported-core-abi] out=$OUT_DIR"
echo "[native-v2-imported-core-abi] program=$PROGRAM"

if [[ ! -f "$PROGRAM" ]]; then
  echo "[native-v2-imported-core-abi] FAIL: missing program $PROGRAM" >&2
  exit 1
fi

run_log() {
  local name="$1"
  shift
  echo "[native-v2-imported-core-abi] running $name"
  "$@" >"$LOG_DIR/$name.log" 2>&1
}

run_log frontend_convergence bash scripts/ci/native_v2_frontend_convergence_gate.sh

run_log prebundle \
  "$SOUC_BIN" run self-hosted/compiler/native_prebundle.sio -- "$PROGRAM" -o "$BUNDLE" --root tests/selfhost/native_runtime

if [[ ! -s "$BUNDLE" ]]; then
  echo "[native-v2-imported-core-abi] FAIL: bundle not produced" >&2
  cat "$LOG_DIR/prebundle.log" >&2 || true
  exit 1
fi

source_markers="$(awk 'BEGIN { n = 0 } /^\/\/ SOURCE:/ { n += 1 } END { print n }' "$BUNDLE")"
if [[ "$source_markers" -ne 2 ]]; then
  echo "[native-v2-imported-core-abi] FAIL: expected 2 SOURCE markers, got $source_markers" >&2
  exit 1
fi

if grep -Eq '^(use |module )' "$BUNDLE"; then
  echo "[native-v2-imported-core-abi] FAIL: bundle still contains use/module lines" >&2
  grep -En '^(use |module )' "$BUNDLE" >&2 || true
  exit 1
fi

if ! grep -q '^// SOURCE: tests/selfhost/native_runtime/import_core_abi/mod.sio$' "$BUNDLE" ||
   ! grep -q '^// SOURCE: tests/selfhost/native_runtime/import_core_abi_42.sio$' "$BUNDLE"; then
  echo "[native-v2-imported-core-abi] FAIL: expected imported and main source markers" >&2
  grep '^// SOURCE:' "$BUNDLE" >&2 || true
  exit 1
fi

run_log native_compile \
  "$SOUC_BIN" run self-hosted/compiler/native_compile_driver.sio -- "$BUNDLE" -o "$ELF"

if [[ ! -f "$ELF" ]]; then
  echo "[native-v2-imported-core-abi] FAIL: native ELF not produced" >&2
  cat "$LOG_DIR/native_compile.log" >&2 || true
  exit 1
fi

chmod +x "$ELF" 2>/dev/null || true

if command -v file >/dev/null 2>&1; then
  file "$ELF" >"$LOG_DIR/import_core_abi_42.file.txt"
  if ! grep -q 'ELF 64-bit LSB executable, x86-64' "$LOG_DIR/import_core_abi_42.file.txt"; then
    echo "[native-v2-imported-core-abi] FAIL: unexpected native artifact kind" >&2
    cat "$LOG_DIR/import_core_abi_42.file.txt" >&2
    exit 1
  fi
fi

set +e
"$ELF" >"$LOG_DIR/import_core_abi_42.stdout" 2>"$LOG_DIR/import_core_abi_42.stderr"
runtime_rc=$?
set -e

if [[ "$runtime_rc" -ne 42 ]]; then
  echo "[native-v2-imported-core-abi] FAIL: expected runtime exit 42, got $runtime_rc" >&2
  cat "$LOG_DIR/import_core_abi_42.stdout" >&2 || true
  cat "$LOG_DIR/import_core_abi_42.stderr" >&2 || true
  exit 1
fi

python3 - "$SUMMARY_JSON" "$SOUC_BIN" "$PROGRAM" "$BUNDLE" "$ELF" "$source_markers" "$OUT_DIR" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

summary_path = pathlib.Path(sys.argv[1])
souc_bin = sys.argv[2]
program = sys.argv[3]
bundle = pathlib.Path(sys.argv[4])
elf = pathlib.Path(sys.argv[5])
source_markers = int(sys.argv[6])
out_dir = pathlib.Path(sys.argv[7])

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "schema": "sounio.native_v2_imported_core_abi.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "status": "pass",
    "compiler_resolved": souc_bin,
    "target": "x86_64-linux",
    "program": program,
    "bundle": str(bundle),
    "native_elf": str(elf),
    "native_elf_sha256": sha256(elf),
    "runtime_exit": 42,
    "fallback_path": "prebundle",
    "host_callback": "none",
    "imported_prebundle_native": True,
    "source_markers": source_markers,
    "artifact_dir": str(out_dir),
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[native-v2-imported-core-abi] PASS: imported core ABI prebundles to native ELF"
echo "[native-v2-imported-core-abi] summary=$SUMMARY_JSON"
