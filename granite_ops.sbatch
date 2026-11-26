#!/usr/bin/env bash
#SBATCH --job-name=granite-tail
#SBATCH --no-requeue
#SBATCH --partition=gpu-orangefs
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-tail-%j.log
set -euo pipefail
work=/tmp/granite-tail-$SLURM_JOB_ID
mkdir -p "$work"
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$1" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; rec "tail-exit-$rc"' EXIT
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt"
export HOME="$work/home" XDG_CACHE_HOME="$work/home/.cache"
mkdir -p "$HOME"
ulimit -s unlimited
cd "$work"
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$1"
git -C src checkout -q --detach FETCH_HEAD
cd src
bash scripts/ci/build_modular_madaros.sh "$work/madaros" > "$work/build.log" 2>&1
rec compiler-built
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
export SOUNIO_MADAROS_BIN="$work/madaros"
./bin/souc check stdlib/llm/qwen3.sio > "$work/check-lib.log" 2>&1
./bin/souc check tests/llm/llm_v2_adversarial_ftz.sio > "$work/check-test.log" 2>&1
"$work/madaros" build tests/llm/llm_v2_adversarial_ftz.sio -o "$work/adv" > "$work/build-adv.log" 2>&1
rec adv-built
SOUNIO_AUDIT_BACKEND=cpu SOUNIO_ADV_OUT="$work/cpu.bin" "$work/adv" > "$work/cpu.log" 2>&1
grep -q GRANITE_ARGMAX_TAIL_CPU_PASS "$work/cpu.log"
rec granite-tail-cpu-pass
if [[ "${2:-cuda}" == cuda ]]; then
  SOUNIO_AUDIT_BACKEND=cuda SOUNIO_ADV_OUT="$work/cuda.bin" "$work/adv" > "$work/cuda.log" 2>&1
  grep -q GRANITE_ARGMAX_TAIL_CUDA_PASS "$work/cuda.log"
  rec granite-tail-cuda-pass
fi
sha256sum "$work/adv" "$work/"*.bin
if [[ -d /orangefs/training/llm-granite-n10 ]]; then
  mkdir -p "/orangefs/training/llm-granite-n10/ops-$SLURM_JOB_ID"
  cp "$work/"*.log "$work/"*.bin "/orangefs/training/llm-granite-n10/ops-$SLURM_JOB_ID/"
fi
exit 0
__CA_BUNDLE__
