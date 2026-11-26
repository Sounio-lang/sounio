#!/usr/bin/env bash
#SBATCH --job-name=granite-runtime-check
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=9G
#SBATCH --gres=none
#SBATCH --time=00:40:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-runtime-check-%j.log
set -uo pipefail
work=/tmp/granite-runtime-check-$SLURM_JOB_ID
mkdir -p "$work"
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt"
ulimit -s unlimited
cd "$work"
set -e
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$1"
git -C src checkout -q --detach FETCH_HEAD
cd src
compiler=/tmp/granite-tail-29222/madaros
if [[ ! -x "$compiler" && -x /tmp/granite-runtime-check-29232/madaros ]]; then compiler=/tmp/granite-runtime-check-29232/madaros; fi
if [[ ! -x "$compiler" ]]; then compiler="$work/madaros"; bash scripts/ci/build_modular_madaros.sh "$compiler" > "$work/build.log" 2>&1; fi
rec "compiler-sha-$(sha256sum "$compiler" | cut -d' ' -f1)"
export SOUNIO_STDLIB_PATH="$PWD/stdlib" SOUNIO_MADAROS_BIN="$compiler"
art=/orangefs/training/llm-granite-n10/cpu-$SLURM_JOB_ID
if [[ -w /orangefs/training/llm-granite-n10 ]]; then mkdir -p "$art"; else art=""; fi
set +e
for f in tests/llm/llm_raw_memory_contract.sio tests/llm/llm_mmap_contract.sio tests/llm/llm_granite_argmax_tail_cpu.sio tests/llm/llm_v2_adversarial_ftz.sio tests/llm/llm_v2_61440_audit.sio; do
  name=$(basename "$f" .sio)
  ./bin/souc check "$f" > "$work/check-$name.log" 2>&1
  rc=$?; rec "check-$name-rc-$rc"
  if [[ $rc != 0 ]]; then
    grep -E 'error\[' "$work/check-$name.log" | head -8 | while IFS= read -r l; do rec "$l"; done
    grep -E '= name' "$work/check-$name.log" | sort -u | head -12 | while IFS= read -r l; do rec "$l"; done
    continue
  fi
  "$compiler" build "$f" -o "$work/$name" > "$work/build-$name.log" 2>&1
  rc=$?; rec "build-$name-rc-$rc"
  if [[ $rc != 0 || ! -f "$work/$name" ]]; then
    tail -10 "$work/build-$name.log" | while IFS= read -r l; do rec "$l"; done
    continue
  fi
  chmod +x "$work/$name"
  rec "sha-$name-$(sha256sum "$work/$name" | cut -d' ' -f1)"
  if [[ -n "$art" ]]; then cp "$work/$name" "$art/$name"; fi
  if [[ "$name" != llm_v2_61440_audit ]]; then
    SOUNIO_AUDIT_BACKEND=cpu SOUNIO_ADV_OUT="$work/cpu.bin" "$work/$name" > "$work/run-$name.log" 2>&1
    rec "run-$name-rc-$?"
    grep -E 'PASS|FAIL|TOTAL_' "$work/run-$name.log" | tail -5 | while IFS= read -r l; do rec "$l"; done
  else
    SOUNIO_AUDIT_BACKEND=cuda "$work/$name" --arch granite --prompts 1 > "$work/cli-shape.log" 2>&1
    rec "registered-shape-refusal-rc-$?"
    grep -E 'AUDIT_|GRANITE_' "$work/cli-shape.log" | head -4 | while IFS= read -r l; do rec "$l"; done
    SOUNIO_AUDIT_BACKEND=cuda "$work/$name" --arch unknown > "$work/cli-arch.log" 2>&1
    rec "unknown-arch-refusal-rc-$?"
    grep -E 'AUDIT_|GRANITE_' "$work/cli-arch.log" | head -4 | while IFS= read -r l; do rec "$l"; done
  fi
done
c++ -std=c++23 -O2 -Wall -Wextra -Werror tools/sounio-llm/granite_prompt_ids.cpp -lpcre2-8 -o "$work/ids" > "$work/build-ids.log" 2>&1
rec "build-ids-rc-$?"
tail -4 "$work/build-ids.log" | while IFS= read -r l; do rec "$l"; done
c++ -std=c++23 -O2 -Wall -Wextra -Werror tools/sounio-llm/cmp_61440.cpp -o "$work/cmp" > "$work/build-cmp.log" 2>&1
rec "build-cmp-rc-$?"
if [[ -x "$work/cmp" ]]; then "$work/cmp" --selftest > "$work/cmp-selftest.log" 2>&1; rec "cmp-selftest-rc-$?"; rec "cmp-sha-$(sha256sum "$work/cmp" | cut -d' ' -f1)"; fi
if [[ -n "$art" ]]; then
  cp "$work/"*.log "$art/" 2>/dev/null || true
  if [[ -s "$work/cpu.bin" ]]; then cp "$work/cpu.bin" "$art/"; fi
  (cd "$art" && find . -maxdepth 1 -type f ! -name "SHA256SUMS*" -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS)
  rec "cpu-artifacts-$art"
fi
exit 0
__CA_BUNDLE__
