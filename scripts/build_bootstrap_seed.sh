#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/debug/souc}"
SEED_PATH="${SEED_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
CACHE_PATH="${CACHE_PATH:-self-hosted/.sounio_bytecode.sobc}"
BUILD_TIMEOUT_SECS="${BUILD_TIMEOUT_SECS:-900}"
SEED_TIMEOUT_SECS="${SEED_TIMEOUT_SECS:-900}"
# Fail closed by default. Local recovery can opt in with SEED_FALLBACK_ENABLED=1.
SEED_FALLBACK_ENABLED="${SEED_FALLBACK_ENABLED:-0}"
TRUSTED_SEED_FALLBACK_PATH="${TRUSTED_SEED_FALLBACK_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
BOOTSTRAP_KERNEL_MANIFEST_PATH="${BOOTSTRAP_KERNEL_MANIFEST_PATH:-${SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST:-bootstrap/selfhost-kernel.manifest}}"
BOOTSTRAP_SEED_TRUSTED_KEY="${SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY:-sounio-dev}"

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
echo "bootstrap_seed_trusted_key=$BOOTSTRAP_SEED_TRUSTED_KEY"

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
      echo "error: failed to compile self-hosted suite while generating seed (fallback disabled; set SEED_FALLBACK_ENABLED=1 to opt in)" >&2
      exit 1
    fi

    if [[ ! -f "$TRUSTED_SEED_FALLBACK_PATH" ]]; then
      echo "error: failed self-hosted seed build and trusted fallback seed missing at $TRUSTED_SEED_FALLBACK_PATH" >&2
      exit 1
    fi

    local_fallback_checksum="${TRUSTED_SEED_FALLBACK_PATH}.sha256"
    local_fallback_sig="${TRUSTED_SEED_FALLBACK_PATH}.sig"
    if [[ ! -f "$local_fallback_checksum" ]]; then
      echo "error: trusted fallback checksum missing at $local_fallback_checksum" >&2
      exit 1
    fi
    if [[ ! -f "$local_fallback_sig" ]]; then
      echo "error: trusted fallback signature missing at $local_fallback_sig" >&2
      exit 1
    fi

    python3 - "$TRUSTED_SEED_FALLBACK_PATH" "$local_fallback_checksum" "$local_fallback_sig" "$BOOTSTRAP_SEED_TRUSTED_KEY" <<'PY'
import hashlib
import pathlib
import re
import sys

seed_path = pathlib.Path(sys.argv[1])
checksum_path = pathlib.Path(sys.argv[2])
sig_path = pathlib.Path(sys.argv[3])
trusted_key = sys.argv[4]

seed_bytes = seed_path.read_bytes()
digest = hashlib.sha256(seed_bytes).hexdigest()

def parse_hex_digest(text: str) -> str | None:
    for raw in text.split():
        token = raw.strip("\"',;()[]{}")
        if not token:
            continue
        candidate = token
        if "=" in token:
            candidate = token.rsplit("=", 1)[1].strip()
        elif ":" in token:
            candidate = token.rsplit(":", 1)[1].strip()
        if len(candidate) == 64 and re.fullmatch(r"[0-9A-Fa-f]{64}", candidate):
            return candidate.lower()
    return None

checksum_text = checksum_path.read_text(encoding="utf-8")
checksum_digest = parse_hex_digest(checksum_text)
if checksum_digest != digest:
    raise SystemExit(
        f"error: trusted fallback checksum mismatch path={checksum_path} expected={digest} actual={checksum_digest}"
    )

sig_text = sig_path.read_text(encoding="utf-8")
signature_tokens = sig_text.split()
if not signature_tokens or signature_tokens[0] != "SOUNIO-SEED-SIG-V1":
    raise SystemExit(
        f"error: trusted fallback signature missing marker SOUNIO-SEED-SIG-V1 path={sig_path}"
    )
sig_key = None
sig_digest = None
for token in signature_tokens[1:]:
    if token.startswith("key="):
        sig_key = token.split("=", 1)[1].strip()
    elif token.startswith("sha256="):
        sig_digest = token.split("=", 1)[1].strip()
if not sig_key:
    raise SystemExit(
        f"error: trusted fallback signature missing key field path={sig_path}"
    )
if sig_key != trusted_key:
    raise SystemExit(
        f"error: trusted fallback signature key mismatch path={sig_path} expected={trusted_key} actual={sig_key}"
    )
if not sig_digest or len(sig_digest) != 64 or not re.fullmatch(r"[0-9A-Fa-f]{64}", sig_digest):
    raise SystemExit(
        f"error: trusted fallback signature digest malformed path={sig_path} actual={sig_digest}"
    )
sig_digest = sig_digest.lower()
if sig_digest != digest:
    raise SystemExit(
        f"error: trusted fallback signature digest mismatch path={sig_path} expected={digest} actual={sig_digest}"
    )
print(f"verified_fallback_seed_sha256={digest}")
PY

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
reserved = struct.unpack_from("<H", seed, 10)[0]
if reserved != 0:
    raise SystemExit(
        f"error: unsupported trusted seed reserved header={reserved}: {seed_path}"
    )

payload_len = struct.unpack_from("<Q", seed, 12)[0]
if payload_len > 16 * 1024 * 1024:
    raise SystemExit(
        f"error: trusted seed payload too large bytes={payload_len} max={16 * 1024 * 1024}: {seed_path}"
    )
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

python3 - "$CACHE_PATH" "$SEED_PATH" "$BOOTSTRAP_SEED_TRUSTED_KEY" <<'PY'
import hashlib
import pathlib
import struct
import sys

cache_path = pathlib.Path(sys.argv[1])
seed_path = pathlib.Path(sys.argv[2])
trusted_key = sys.argv[3]

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
    f"SOUNIO-SEED-SIG-V1 key={trusted_key} sha256={digest}\n",
    encoding="utf-8",
)
print(f"seed={seed_path}")
print(f"sha256={digest}")
PY

echo "BUILD_BOOTSTRAP_SEED_DONE seed=$SEED_PATH"
