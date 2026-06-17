#!/usr/bin/env bash
# Erdős90 Option-B: K-AXI HC smoke GPU launch on cluster (opt-in).
#
# Local preflight: CPU oracle + warp parity + PTX emit; optional sbatch when
# SOUNIO_ERDOS90_KAXI_GPU_SMOKE_RUN_SLURM=1 and sbatch is available.
#
# Usage:
#   slurm-jobs/erdos90/submit_kaxi_gpu_smoke.sh
#   SOUNIO_ERDOS90_KAXI_GPU_SMOKE_RUN_SLURM=1 slurm-jobs/erdos90/submit_kaxi_gpu_smoke.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"

SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
KRETIKOS="${KRETIKOS:-$ROOT_DIR/bin/kretikos}"
if [[ "$SOUC" == "$ROOT_DIR/bin/souc" && -z "${SOUNIO_SOUC_ENGINE:-}" && -x "$ROOT_DIR/bin/souc-lean-single-x86_64" ]]; then
  SOUC="$ROOT_DIR/bin/souc-lean-single-x86_64"
fi
export SOUNIO_KRETIKOS_COMPILER="${SOUNIO_KRETIKOS_COMPILER:-$SOUC}"
OUT_DIR="${SOUNIO_ERDOS90_KAXI_GPU_SMOKE_OUT:-$ROOT_DIR/artifacts/erdos90/kaxi_gpu_smoke}"
mkdir -p "$OUT_DIR"

echo "[erdos90-kaxi-gpu-smoke-slurm] local gate"
bash "$ROOT_DIR/scripts/gates/erdos90_subset400_kaxi_gpu_smoke_gate.sh" \
  >"$OUT_DIR/local_gate.log" 2>&1
grep -q 'PASS' "$OUT_DIR/local_gate.log"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[erdos90-kaxi-gpu-smoke-slurm] emit PTX"
"$KRETIKOS" kaxi-emit-ptx erdos90_hc_smoke -o "$OUT_DIR/erdos90_hc_smoke.ptx" \
  >"$OUT_DIR/emit.log" 2>&1

echo "[erdos90-kaxi-gpu-smoke-slurm] pack init from CPU oracle"
"$SOUC" stdlib/research/erdos90_kaxi_hc_smoke.sio "$WORK/smoke.elf" >/dev/null
chmod +x "$WORK/smoke.elf"
"$WORK/smoke.elf" >"$WORK/smoke.log" 2>&1
MEM_WORDS="$(grep '^KAXI_SMOKE_MEM_WORDS=' "$WORK/smoke.log" | sed 's/^KAXI_SMOKE_MEM_WORDS=//')"
THREADS=8

python3 - "$WORK/smoke.log" "$OUT_DIR/init_mem.bin" "$MEM_WORDS" <<'PY'
import pathlib
import re
import struct
import sys

log = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(sys.argv[2])
words = int(sys.argv[3])
start = log.find("KAXI_SMOKE_INIT_CSV=")
if start < 0:
    raise SystemExit("KAXI_SMOKE_INIT_CSV missing from smoke log")
end = log.find("KAXI_SMOKE_DONE", start)
chunk = log[start:end if end >= 0 else None]
chunk = chunk.split("=", 1)[1]
vals = [int(x) for x in re.findall(r"-?\d+", chunk)]
if len(vals) != words:
    raise SystemExit(f"init word count mismatch: got {len(vals)} want {words}")
out.write_bytes(struct.pack("<" + "q" * words, *vals))
PY

STAGE="$OUT_DIR/stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$OUT_DIR/erdos90_hc_smoke.ptx" "$STAGE/kernel.ptx"
cp "$OUT_DIR/init_mem.bin" "$STAGE/init_mem.bin"
cp "$ROOT_DIR/scripts/gpu/kaxi_ptx_runner.c" "$STAGE/"
cc -O2 "$STAGE/kaxi_ptx_runner.c" -ldl -lm -o "$STAGE/runner"

cat >"$STAGE/run.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./runner kernel.ptx \
  --mode basic --threads 8 --mem-words 437 \
  --init-file init_mem.bin --type i64 --print-count 437 \
  > gpu.log 2>&1
grep -q 'sounio_kaxi_runtime status=pass' gpu.log
EOS
chmod +x "$STAGE/run.sh"
tar -C "$STAGE" -czf "$OUT_DIR/erdos90_kaxi_gpu_smoke.tgz" .

if [[ "${SOUNIO_ERDOS90_KAXI_GPU_SMOKE_RUN_SLURM:-0}" != "1" ]]; then
  echo "[erdos90-kaxi-gpu-smoke-slurm] SKIPPED slurm (set SOUNIO_ERDOS90_KAXI_GPU_SMOKE_RUN_SLURM=1)"
  echo "[erdos90-kaxi-gpu-smoke-slurm] artifact=$OUT_DIR/erdos90_kaxi_gpu_smoke.tgz"
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[erdos90-kaxi-gpu-smoke-slurm] SKIPPED sbatch_missing" >&2
  exit 0
fi

PARTITION="${SOUNIO_ERDOS90_KAXI_GPU_SMOKE_PARTITION:-gpu-orangefs}"
GRES="${SOUNIO_ERDOS90_KAXI_GPU_SMOKE_GRES:-gpu:1}"
JOB_MEM="${SOUNIO_ERDOS90_KAXI_GPU_SMOKE_MEM:-4G}"
JOB_TIME="${SOUNIO_ERDOS90_KAXI_GPU_SMOKE_TIME:-00:05:00}"
NODELIST="${SOUNIO_ERDOS90_KAXI_GPU_SMOKE_NODELIST:-}"

JOB="$OUT_DIR/job.sh"
cat >"$JOB" <<EOS
#!/usr/bin/env bash
#SBATCH --job-name=erdos90-kaxi-smoke
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
#SBATCH --output=${OUT_DIR}/job_%j.out
#SBATCH --error=${OUT_DIR}/job_%j.err
${NODELIST:+#SBATCH --nodelist=${NODELIST}}

set -euo pipefail
STAGE="\$SLURM_TMPDIR/erdos90_kaxi_smoke"
mkdir -p "\$STAGE"
tar -xzf "${OUT_DIR}/erdos90_kaxi_gpu_smoke.tgz" -C "\$STAGE"
cd "\$STAGE"
./run.sh
EOS
chmod +x "$JOB"

JOB_ID="$(sbatch --parsable "$JOB")"
echo "[erdos90-kaxi-gpu-smoke-slurm] submitted job=$JOB_ID artifact=$OUT_DIR/erdos90_kaxi_gpu_smoke.tgz"