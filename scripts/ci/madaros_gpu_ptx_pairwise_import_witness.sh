#!/usr/bin/env bash
# madaros_gpu_ptx_pairwise_import_witness.sh - cheap GPU/PTX module import matrix.
#
# This guards the normal module graph for the PTX lowering spine:
#   gpu::kernel_ir::* + gpu::lower_to_ptx::* + gpu::ptx::*
#
# The historical failure was a check-time privacy/shape explosion in this exact
# combination. Keep this witness small: it only typechecks empty check-only
# probes and does not require CUDA, PTX assembly, or native GPU execution.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[madaros-gpu-ptx-import] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[madaros-gpu-ptx-import] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros

OUT_DIR="${SOUNIO_MADAROS_GPU_PTX_IMPORT_DIR:-$(mktemp -d /tmp/sounio-madaros-gpu-ptx-import.XXXXXX)}"
PROBE_DIR="$OUT_DIR/probes"
LOG_DIR="$OUT_DIR/logs"
RESULTS_TSV="$OUT_DIR/results.tsv"

mkdir -p "$PROBE_DIR" "$LOG_DIR"

echo "[madaros-gpu-ptx-import] START"
echo "[madaros-gpu-ptx-import] madaros=$MADAROS_BIN"
echo "[madaros-gpu-ptx-import] out=$OUT_DIR"

write_probe() {
  local case_id="$1"
  shift

  local probe="$PROBE_DIR/$case_id.sio"
  {
    echo '//@ check-only'
    for module_name in "$@"; do
      printf 'use gpu::%s::*\n' "$module_name"
    done
    echo
    echo 'fn main() -> i32 {'
    echo '    0'
    echo '}'
  } >"$probe"

  echo "$probe"
}

run_case() {
  local case_id="$1"
  shift

  local probe check_log rc status e175 e177 e046
  probe="$(write_probe "$case_id" "$@")"
  check_log="$LOG_DIR/$case_id.check.log"

  set +e
  "$MADAROS_BIN" check "$probe" >"$check_log" 2>&1
  rc=$?
  set -e

  if [[ "$rc" == 0 ]]; then
    status="check_ok"
    printf '%s\t%s\t%s\t%s\n' "$case_id" "$probe" "$rc" "$status" >>"$RESULTS_TSV"
    echo "[madaros-gpu-ptx-import] PASS: $case_id"
    return 0
  fi

  status="check_failed"
  printf '%s\t%s\t%s\t%s\n' "$case_id" "$probe" "$rc" "$status" >>"$RESULTS_TSV"
  e175="$(grep -c 'error\[E175' "$check_log" || true)"
  e177="$(grep -c 'error\[E177' "$check_log" || true)"
  e046="$(grep -c 'error\[E046' "$check_log" || true)"
  echo "[madaros-gpu-ptx-import] FAIL: $case_id rc=$rc E175=$e175 E177=$e177 E046=$e046" >&2
  tail -n 60 "$check_log" >&2 || true
  return 1
}

printf 'case_id\tprobe\trc\tstatus\n' >"$RESULTS_TSV"

run_case kernel_ir__lower_to_ptx kernel_ir lower_to_ptx
run_case lower_to_ptx__ptx lower_to_ptx ptx
run_case kernel_ir__ptx kernel_ir ptx
run_case kernel_ir__lower_to_ptx__ptx kernel_ir lower_to_ptx ptx

echo "[madaros-gpu-ptx-import] PASS: 4/4 import witnesses"
echo "[madaros-gpu-ptx-import] results=$RESULTS_TSV"
