#!/usr/bin/env bash
#SBATCH --job-name=granite-tokenizer-smoke
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --gres=none
#SBATCH --time=00:20:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-tokenizer-smoke-%j.log
set -euo pipefail
work=/tmp/granite-tokenizer-smoke-$SLURM_JOB_ID
mkdir -p "$work"
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; rec "tokenizer-exit-$rc"' EXIT
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt"
ulimit -s unlimited
cd "$work"
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$1"
git -C src checkout -q --detach FETCH_HEAD
cd src
compiler=/tmp/granite-runtime-check-29243/madaros
if [[ ! -x "$compiler" ]]; then compiler="$work/madaros"; bash scripts/ci/build_modular_madaros.sh "$compiler" > "$work/build.log" 2>&1; fi
export SOUNIO_STDLIB_PATH="$PWD/stdlib" SOUNIO_MADAROS_BIN="$compiler"
./bin/souc check tests/llm/llm_granite_tokenizer_smoke.sio > "$work/check.log" 2>&1
rec tokenizer-check-ok
"$compiler" build tests/llm/llm_granite_tokenizer_smoke.sio -o "$work/smoke" > "$work/build-smoke.log" 2>&1
chmod +x "$work/smoke"
rec "tokenizer-smoke-bin-$(sha256sum "$work/smoke" | cut -d' ' -f1)"
[[ -s "$2" && $(stat -c '%s' "$2") -gt 1000000 ]]
rec "tokenizer-json-$(sha256sum "$2" | cut -d' ' -f1)"
set +e
"$work/smoke" "$2" > "$work/run.log" 2>&1
rc=$?
rec "smoke-rc-$rc"
grep -E 'GRANITE_TOKENIZER' "$work/run.log" | while IFS= read -r l; do rec "$l"; done
exit "$rc"
__CA_BUNDLE__
