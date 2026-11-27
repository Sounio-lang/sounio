#!/usr/bin/env bash
#SBATCH --job-name=llm-cuda-alloc-refusal
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/llm-cuda-alloc-refusal-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
work=/tmp/llm-cuda-alloc-refusal-$SLURM_JOB_ID
art=$stage/alloc-refusal-$SLURM_JOB_ID
mkdir -p "$work" "$art"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; cp "$work/"*.log "$art/" 2>/dev/null || true; if ((rc)); then for f in "$work/check.log" "$work/build-probe.log" "$work/build-mock.log"; do if [[ -s "$f" ]]; then tail -8 "$f" | while IFS= read -r l; do rec "$l"; done; fi; done; fi; rec "alloc-refusal-job-exit-$rc"' EXIT
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt" ZIG_GLOBAL_CACHE_DIR="$work/cache" ZIG_LOCAL_CACHE_DIR="$work/local-cache"
export HOME="$work/home" XDG_CACHE_HOME="$work/home/.cache"
mkdir -p "$HOME"
ulimit -s unlimited
cd "$work"
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$1"
git -C src checkout -q --detach FETCH_HEAD
[[ $(git -C src rev-parse HEAD) == "$1" ]]
rec "source-$1"
cd src
bash scripts/ci/build_modular_madaros.sh "$work/madaros" > "$work/build-madaros.log" 2>&1
rec "madaros-sha-$(sha256sum "$work/madaros" | cut -d' ' -f1)"
export SOUNIO_STDLIB_PATH="$PWD/stdlib" SOUNIO_MADAROS_BIN="$work/madaros"
./bin/souc check tests/llm/cuda_init_alloc_refusal.sio > "$work/check.log" 2>&1
"$work/madaros" build tests/llm/cuda_init_alloc_refusal.sio -o "$work/probe" > "$work/build-probe.log" 2>&1
chmod +x "$work/probe"
"$stage/tooling/zig-x86_64-linux-0.15.2/zig" c++ -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC tests/llm/cuda_init_alloc_refusal_mock.cpp -o "$work/libmockcuda.so" > "$work/build-mock.log" 2>&1
head -c64 /dev/zero > "$work/shard.bin"
rc=0
timeout -k 2s 30s "$work/probe" "$work/shard.bin" "$work/libmockcuda.so" > "$work/run.log" 2>&1 || rc=$?
while IFS= read -r l; do rec "$l"; done < "$work/run.log"
rec "probe-rc-$rc"
sha256sum "$work/madaros" "$work/probe" "$work/libmockcuda.so" "$work/run.log" > "$art/SHA256SUMS"
[[ "$rc" -eq 0 ]] && grep -qx RUNNER_REFUSAL_PASS "$work/run.log"
if [[ "${2:-after}" == before ]]; then
  ! grep -q CUDA_DEVICE_ALLOC_REFUSED "$work/run.log"
  [[ $(grep -c '^FAKE_ALLOC_CALL=' "$work/run.log") -gt 2 ]]
  rec BASELINE_SILENT_CONTINUATION_REPRODUCED
else
  grep -qx 'CUDA_DEVICE_ALLOC_REFUSED stage=shard bytes=64 CUresult=2' "$work/run.log"
  [[ $(grep -c '^FAKE_ALLOC_CALL=' "$work/run.log") -eq 2 ]]
  grep -qx 'FAKE_CTX_DESTROY allocations=2 frees=1' "$work/run.log"
  rec CUDA_INIT_ALLOC_REFUSAL_PASS
fi
exit 0
__CA_BUNDLE__
