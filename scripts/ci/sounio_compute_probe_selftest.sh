#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$ROOT_DIR/scripts/dev/sounio_compute_probe.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cat >"$TMP_DIR/slurm-ready.txt" <<'EOF'
all|up|1|(null)|gpuorangefs-multi|idle
gpu-orangefs*|up|2|gpu:1|gpuorangefs|mix
EOF
printf 'dl380-proxmox|True|1\n' >"$TMP_DIR/u250-ready.txt"
printf 'dl380-proxmox|Unknown|1\n' >"$TMP_DIR/u250-unknown.txt"
printf 'full-compiler current\n' >"$TMP_DIR/beagle-ready.txt"
printf 'HTTP 401 Unauthorized: token expired\n' >"$TMP_DIR/beagle-blocked.txt"

assert_field() {
  local output="$1"
  local field="$2"
  local value="$3"
  grep -Fxq "$field: $value" <<<"$output" || {
    echo "expected '$field: $value' in:" >&2
    echo "$output" >&2
    exit 1
  }
}

run_slurm() {
  SOUNIO_COMPUTE_PROBE_SLURM_FIXTURE="$TMP_DIR/slurm-ready.txt" \
    bash "$PROBE" --class "$1"
}

for class in compiler.cpu-heavy compiler.gpu research.batch; do
  output="$(run_slurm "$class")"
  assert_field "$output" classification READY
  assert_field "$output" submission_performed no
done

output="$(SOUNIO_COMPUTE_PROBE_K8S_FIXTURE="$TMP_DIR/u250-ready.txt" \
  bash "$PROBE" --class compiler.fpga-u250)"
assert_field "$output" classification READY
assert_field "$output" submission_performed no

output="$(SOUNIO_COMPUTE_PROBE_K8S_FIXTURE="$TMP_DIR/u250-unknown.txt" \
  bash "$PROBE" --class compiler.fpga-u250)"
assert_field "$output" classification BLOCKED
assert_field "$output" reason no_ready_u250_allocatable_node

output="$(SOUNIO_COMPUTE_PROBE_BEAGLE_FIXTURE="$TMP_DIR/beagle-ready.txt" \
  bash "$PROBE" --class foundry.release)"
assert_field "$output" classification READY

output="$(SOUNIO_COMPUTE_PROBE_BEAGLE_FIXTURE="$TMP_DIR/beagle-blocked.txt" \
  SOUNIO_COMPUTE_PROBE_BEAGLE_RC=1 bash "$PROBE" --class foundry.release)"
assert_field "$output" classification BLOCKED
assert_field "$output" reason beagle_auth_or_profile_unavailable

if bash "$PROBE" --class unknown >/dev/null 2>&1; then
  echo "unknown class unexpectedly succeeded" >&2
  exit 1
fi

echo SOUNIO_COMPUTE_PROBE_SELFTEST_OK
