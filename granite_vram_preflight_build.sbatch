#!/usr/bin/env bash
#SBATCH --job-name=granite-vram-preflight-build
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --time=00:10:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-vram-preflight-build-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
work=/tmp/granite-vram-preflight-build-$SLURM_JOB_ID
art=$stage/vram-preflight-$SLURM_JOB_ID
mkdir -p "$work" "$art"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$1" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; cp "$work/"*.log "$art/" 2>/dev/null || true; if ((rc)) && [[ -s "$work/build.log" ]]; then tail -8 "$work/build.log" | while IFS= read -r l; do rec "$(printf "%s" "$l" | cut -c1-120)"; done; fi; rec "preflight-build-exit-$rc"' EXIT
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt" ZIG_GLOBAL_CACHE_DIR="$work/cache" ZIG_LOCAL_CACHE_DIR="$work/local-cache"
cd "$work"
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$1"
git -C src checkout -q --detach FETCH_HEAD
[[ $(git -C src rev-parse HEAD) == "$1" ]]
"$stage/tooling/zig-x86_64-linux-0.15.2/zig" c++ -std=c++23 -O2 -Wall -Wextra -Werror src/tools/sounio-llm/granite_vram_preflight.cpp -ldl -o "$art/vram_preflight" > "$work/build.log" 2>&1
"$art/vram_preflight" --selftest | tee "$work/selftest.log"
grep -qx VRAM_ADMISSION_SELFTEST_PASS "$work/selftest.log"
printf '%s\n' "$1" > "$art/SOURCE_COMMIT"
(cd "$art" && sha256sum vram_preflight SOURCE_COMMIT > SHA256SUMS)
rec "vram-preflight-sha-$(sha256sum "$art/vram_preflight" | cut -d' ' -f1)"
rec VRAM_ADMISSION_SELFTEST_PASS
exit 0
__CA_BUNDLE__
