#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/debug/souc}"
SEED_PATH="${SEED_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
CACHE_PATH="${CACHE_PATH:-self-hosted/.sounio_bytecode.sobc}"
BUILD_TIMEOUT_SECS="${BUILD_TIMEOUT_SECS:-900}"
SEED_TIMEOUT_SECS="${SEED_TIMEOUT_SECS:-900}"
SEED_FALLBACK_ENABLED="${SEED_FALLBACK_ENABLED:-1}"
TRUSTED_SEED_FALLBACK_PATH="${TRUSTED_SEED_FALLBACK_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
BOOTSTRAP_KERNEL_MANIFEST_PATH="${BOOTSTRAP_KERNEL_MANIFEST_PATH:-${SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST:-bootstrap/selfhost-kernel.manifest}}"

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
echo "bootstrap_kernel_manifest=$BOOTSTRAP_KERNEL_MANIFEST_PATH"

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
  SOUNIO_SELFHOST_STRICT_MODULE_GATING="1" \
  SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST="$BOOTSTRAP_KERNEL_MANIFEST_PATH" \
  SOUNIO_SELFHOST_PIPELINE="driver" \
  SOUNIO_SELFHOST_WRITE_DIR_CACHE="1" \
  SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT="0" \
  "$SOUC_BIN" run self-hosted/ -- parse-all shard 0 1 balanced >/tmp/sounio-seed-build.log 2>&1 || {
    cat /tmp/sounio-seed-build.log >&2 || true
    if [[ "$SEED_FALLBACK_ENABLED" != "1" ]]; then
      echo "error: failed to compile self-hosted suite while generating seed" >&2
      exit 1
    fi

    if [[ ! -f "$TRUSTED_SEED_FALLBACK_PATH" ]]; then
      echo "error: failed self-hosted seed build and trusted fallback seed missing at $TRUSTED_SEED_FALLBACK_PATH" >&2
      exit 1
    fi

    echo "warning: self-hosted seed build failed; extracting payload from trusted seed fallback: $TRUSTED_SEED_FALLBACK_PATH" >&2
    python3 - "$TRUSTED_SEED_FALLBACK_PATH" "$CACHE_PATH" <<'PY'
import pathlib
import struct
import sys

seed_path = pathlib.Path(sys.argv[1])
cache_path = pathlib.Path(sys.argv[2])
seed = seed_path.read_bytes()

if len(seed) < 20:
    raise SystemExit(f"error: trusted seed too small: {seed_path}")
if seed[:8] != b"SNSDSEED":
    raise SystemExit(f"error: trusted seed has invalid magic: {seed_path}")

version = struct.unpack_from("<H", seed, 8)[0]
if version != 1:
    raise SystemExit(f"error: unsupported trusted seed version={version}: {seed_path}")

payload_len = struct.unpack_from("<Q", seed, 12)[0]
payload = seed[20:20 + payload_len]
if len(payload) != payload_len:
    raise SystemExit(
        f"error: trusted seed payload truncated: expected={payload_len} got={len(payload)}"
    )
cache_path.parent.mkdir(parents=True, exist_ok=True)
cache_path.write_bytes(payload)
print(f"fallback_cache={cache_path}")
print(f"payload_len={payload_len}")
PY
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
