#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/debug/souc}"
SEED_PATH="${SEED_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
CACHE_PATH="${CACHE_PATH:-self-hosted/.sounio_bytecode.sobc}"
BUILD_TIMEOUT_SECS="${BUILD_TIMEOUT_SECS:-900}"
SEED_TIMEOUT_SECS="${SEED_TIMEOUT_SECS:-900}"

run_with_timeout() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
    return $?
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$seconds" "$@" <<'PY'
import subprocess
import sys

seconds = int(sys.argv[1])
command = sys.argv[2:]
try:
    completed = subprocess.run(command, timeout=seconds)
    sys.exit(completed.returncode)
except subprocess.TimeoutExpired:
    sys.exit(124)
PY
    return $?
  fi

  "$@"
}

echo "BUILD_BOOTSTRAP_SEED_START"
echo "seed_path=$SEED_PATH"
echo "cache_path=$CACHE_PATH"

mkdir -p "$(dirname "$SEED_PATH")"
rm -f "$CACHE_PATH"

run_with_timeout "$BUILD_TIMEOUT_SECS" cargo build -p souc

if [ ! -x "$SOUC_BIN" ]; then
  echo "error: missing compiler binary at $SOUC_BIN" >&2
  exit 1
fi

# Compile the self-hosted suite and force writing deterministic directory cache bytes.
run_with_timeout "$SEED_TIMEOUT_SECS" env \
  SOUNIO_BOOTSTRAP_SEED_ENFORCE="0" \
  SOUNIO_BOOTSTRAP_SEED_PATH="/tmp/sounio-seed-build-missing-${PPID}-${BASHPID}.sio.bin" \
  SOUNIO_SELFHOST_PIPELINE="driver" \
  SOUNIO_SELFHOST_WRITE_DIR_CACHE="1" \
  SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT="0" \
  "$SOUC_BIN" run self-hosted/ -- parse-all shard 0 1 balanced >/tmp/sounio-seed-build.log 2>&1 || {
    cat /tmp/sounio-seed-build.log >&2 || true
    echo "error: failed to compile self-hosted suite while generating seed" >&2
    exit 1
  }

if [ ! -f "$CACHE_PATH" ]; then
  echo "error: missing bytecode cache at $CACHE_PATH" >&2
  exit 1
fi

python3 - "$CACHE_PATH" "$SEED_PATH" <<'PY'
import hashlib
import pathlib
import struct
import sys

cache_path = pathlib.Path(sys.argv[1])
seed_path = pathlib.Path(sys.argv[2])

payload = cache_path.read_bytes()
if not payload:
    raise SystemExit(f"error: empty payload cache: {cache_path}")

magic = b"SNSDSEED"
version = 1
reserved = 0
header = magic + struct.pack("<H", version) + struct.pack("<H", reserved) + struct.pack("<Q", len(payload))
seed_bytes = header + payload
seed_path.write_bytes(seed_bytes)

digest = hashlib.sha256(seed_bytes).hexdigest()
seed_path.with_suffix(seed_path.suffix + ".sha256").write_text(f"{digest}  {seed_path.name}\n", encoding="utf-8")
seed_path.with_suffix(seed_path.suffix + ".sig").write_text(
    f"SOUNIO-SEED-SIG-V1 key=sounio-dev sha256={digest}\n",
    encoding="utf-8",
)
print(f"seed={seed_path}")
print(f"sha256={digest}")
PY

echo "BUILD_BOOTSTRAP_SEED_DONE seed=$SEED_PATH"
