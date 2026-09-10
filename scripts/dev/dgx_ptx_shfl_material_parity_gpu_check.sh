#!/usr/bin/env bash
# ADR-009 empirical companion: compiles and runs the shfl.sync.bfly.b32
# dump probe on a real DGX Spark GPU, then feeds the observed output
# matrix to the Futhark verified_foreign_reference oracle's `check`
# entry for a genuine hardware-vs-specification comparison.
#
# Best-effort: if the DGX Spark host is unreachable (no LAN access,
# CI runner, etc.), this reports SKIP rather than FAIL -- the
# algebraic self-check in dgx_ptx_shfl_material_parity_futhark_gate.sh
# remains the hard-failing default gate.
#
# Last manually verified: 2026-09-04, NVIDIA GB10 (compute_capability
# 12.1), 0/256 mismatches, involution_check=true.

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DGX_SPARK_HOST="${DGX_SPARK_HOST:-192.168.3.43}"
DGX_SPARK_USER="${DGX_SPARK_USER:-demetrios}"
DGX_SPARK_TARGET="${DGX_SPARK_USER}@${DGX_SPARK_HOST}"
DGX_SPARK_NVCC="${DGX_SPARK_NVCC:-/usr/local/cuda-13.0/bin/nvcc}"
DGX_SPARK_ARCH="${DGX_SPARK_ARCH:-sm_121}"
DUMP_SOURCE="$ROOT_DIR/tools/pireus/dgx_ptx_shfl_material_parity_dump.cu"
ORACLE_SOURCE="$ROOT_DIR/tools/pireus/dgx_ptx_shfl_material_parity.fut"

skip() {
  printf 'dgx-ptx-shfl-material-parity-gpu-check: SKIP: %s\n' "$*" >&2
  exit 0
}

fail() {
  printf 'dgx-ptx-shfl-material-parity-gpu-check: FAIL: %s\n' "$*" >&2
  exit 1
}

command -v futhark >/dev/null 2>&1 || skip "futhark not on PATH"
[[ -r "$DUMP_SOURCE" ]] || fail "dump probe source not found: $DUMP_SOURCE"
[[ -r "$ORACLE_SOURCE" ]] || fail "oracle source not found: $ORACLE_SOURCE"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)

ssh "${SSH_OPTS[@]}" "$DGX_SPARK_TARGET" true 2>/dev/null || \
  skip "DGX Spark host unreachable: $DGX_SPARK_TARGET"

work="$(mktemp -d "${TMPDIR:-/tmp}/dgx-ptx-shfl-gpu-check.XXXXXX")"
trap 'rm -rf "$work"' EXIT

remote_dump="/tmp/sounio-adr009-shfl-dump-$$.cu"
remote_bin="/tmp/sounio-adr009-shfl-dump-$$"

scp "${SSH_OPTS[@]}" "$DUMP_SOURCE" "$DGX_SPARK_TARGET:$remote_dump" >/dev/null 2>&1 || \
  fail "scp of dump probe failed"

ssh "${SSH_OPTS[@]}" "$DGX_SPARK_TARGET" \
  "$DGX_SPARK_NVCC -arch=$DGX_SPARK_ARCH -o $remote_bin $remote_dump" \
  > "$work/compile.log" 2>&1 || fail "nvcc compile failed: $(cat "$work/compile.log")"

ssh "${SSH_OPTS[@]}" "$DGX_SPARK_TARGET" "$remote_bin; rc=\$?; rm -f $remote_dump $remote_bin; exit \$rc" \
  > "$work/gpu_output.txt" 2>"$work/gpu_stderr.txt" || \
  fail "GPU probe execution failed: $(cat "$work/gpu_stderr.txt")"

gpu_props="$(sed -n '1p' "$work/gpu_stderr.txt")"

python3 - "$work/gpu_output.txt" "$work/futhark_input.txt" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
rows = []
with open(src) as f:
    for line in f:
        vals = line.split()
        if vals:
            rows.append("[" + ",".join(v + "u64" for v in vals) + "]")
with open(dst, "w") as f:
    f.write("[" + ",".join(rows) + "]")
PY

futhark c "$ORACLE_SOURCE" -o "$work/oracle" >/dev/null 2>&1 || fail "futhark compilation failed"
result="$("$work/oracle" -e check < "$work/futhark_input.txt")"
mismatches="$(printf '%s\n' "$result" | sed -n '1p' | tr -d 'i64')"
involution="$(printf '%s\n' "$result" | sed -n '2p')"

[[ "$involution" == "true" ]] || fail "involution self-check failed"
[[ "$mismatches" == "0" ]] || fail "GPU output diverges from Futhark reference in $mismatches/256 cells"

printf 'PIREUS_DGX_PTX_SHFL_MATERIAL_PARITY_GPU_CHECK_V1\n'
printf 'oracle_class=verified_foreign_reference\n'
printf 'producer_language=Futhark\n'
printf 'gpu=%s\n' "$gpu_props"
printf 'mismatched_cells=%s\n' "$mismatches"
printf 'involution_check=PASS\n'
printf 'result=PASS\n'
