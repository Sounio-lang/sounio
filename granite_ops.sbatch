#!/usr/bin/env bash
#SBATCH --job-name=granite-tail
#SBATCH --no-requeue
#SBATCH --comment=lane:llm-granite
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
stage=/orangefs/training/llm-granite-n10
art="$stage/ops-$SLURM_JOB_ID"
trap 'rc=$?; rec "tail-exit-$rc"; if [[ -w "$stage" ]]; then mkdir -p "$art"; cp "$work/"*.log "$work/"*.bin "$work/adv" "$art/" 2>/dev/null || true; (cd "$art" && find . -maxdepth 1 -type f ! -name "SHA256SUMS*" -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS); fi' EXIT
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt"
export HOME="$work/home" XDG_CACHE_HOME="$work/home/.cache"
mkdir -p "$HOME"
ulimit -s unlimited
if [[ -s "$stage/granite-3.1-8b-instruct/SHA256SUMS" ]]; then rec "shared-model-manifest-visible"; else rec "shared-model-manifest-missing"; fi
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
chmod +x "$work/adv"
rec adv-built
SOUNIO_AUDIT_BACKEND=cpu SOUNIO_ADV_OUT="$work/cpu.bin" "$work/adv" > "$work/cpu.log" 2>&1
grep -q GRANITE_ARGMAX_TAIL_CPU_PASS "$work/cpu.log"
rec granite-tail-cpu-pass
if [[ "${2:-cuda}" == cuda ]]; then
  # AFTZ: 32-byte header, eleven (16-byte metadata + name + f32 array) records.
  expected=$((32 + 11 * 16 + 13312 * 4))
  for name in adv/mul_subnorm adv/fma_subnorm adv/add_cancel_subnorm adv/div_subnorm adv/exp_subnorm adv/rsqrt_flt_min adv/gemv_fma_subnorm adv/rmsnorm_tiny adv/div_ties adv/div_subnorm_bound adv/sqrt_ties_pow4_fltmin; do
    expected=$((expected + ${#name}))
  done
  complete_tail() {
    grep -qx GRANITE_ARGMAX_TAIL_CUDA_PASS "$work/cuda.log" &&
    grep -q '^LLM_V2_ADVERSARIAL_FTZ_PASS backend=cuda fma_hits=[1-9][0-9]* subnormals=0$' "$work/cuda.log" &&
    [[ -s "$work/cuda.bin" && $(stat -c %s "$work/cuda.bin") -eq $expected ]] &&
    [[ $(od -An -tx1 -N16 "$work/cuda.bin" | tr -d ' \n') == 4146545a01000000030000000b000000 ]]
  }
  SOUNIO_AUDIT_BACKEND=cuda SOUNIO_ADV_OUT="$work/cuda.bin" "$work/adv" > "$work/cuda.log" 2>&1 &
  pid=$!; rc=0; workaround=0
  while kill -0 "$pid" 2>/dev/null; do
    if complete_tail 2>/dev/null; then
      sleep 10
      if kill -0 "$pid" 2>/dev/null && complete_tail; then
        grep -E '^(Name|State|Threads):' "/proc/$pid/status" > "$work/exit-hang-status.log" 2>/dev/null || true
        kill -KILL "$pid"
        workaround=1
        break
      fi
    fi
    sleep 5
  done
  wait "$pid" || rc=$?
  if (( workaround )); then
    [[ "$rc" -eq 137 ]] && complete_tail || exit 1
    rec r-exit-hang-workaround
  else
    [[ "$rc" -eq 0 ]] && complete_tail || exit 1
  fi
  rec granite-tail-cuda-pass
fi
sha256sum "$work/adv" "$work/"*.bin
exit 0
__CA_BUNDLE__
