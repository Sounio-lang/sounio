#!/usr/bin/env bash
#SBATCH --job-name=granite-diag-build
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-diag-build-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
art=$stage/diag-build-$SLURM_JOB_ID
work=/tmp/granite-diag-build-$SLURM_JOB_ID
mkdir -p "$work" "$art"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; cp "$work/"*.log "$art/" 2>/dev/null || true; if ((rc)); then for f in "$work/check-lib.log" "$work/check-audit.log" "$work/build-audit.log" "$work/build-madaros.log"; do if [[ -s "$f" ]]; then rec "failure-$(basename "$f")"; tail -8 "$f" | while IFS= read -r l; do rec "$l"; done; fi; done; fi; rec "diag-build-exit-$rc"' EXIT
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$0" > "$work/ca.crt"
export GIT_SSL_CAINFO="$work/ca.crt" HOME="$work/home" XDG_CACHE_HOME="$work/home/.cache"
mkdir -p "$HOME"
ulimit -s unlimited
cd "$work"
git init -q src
git -C src fetch -q --depth 1 https://github.com/Sounio-lang/sounio.git "$1"
git -C src checkout -q --detach FETCH_HEAD
[[ $(git -C src rev-parse HEAD) == "$1" ]]
rec "diag-source-$1"
cd src
bash scripts/ci/build_modular_madaros.sh "$work/madaros" > "$work/build-madaros.log" 2>&1
export SOUNIO_STDLIB_PATH="$PWD/stdlib" SOUNIO_MADAROS_BIN="$work/madaros"
./bin/souc check stdlib/llm/hal.sio > "$work/check-hal.log" 2>&1
./bin/souc check stdlib/llm/qwen3.sio > "$work/check-lib.log" 2>&1
./bin/souc check tests/llm/llm_v2_61440_audit.sio > "$work/check-audit.log" 2>&1
"$work/madaros" build tests/llm/llm_v2_61440_audit.sio -o "$work/audit" > "$work/build-audit.log" 2>&1
chmod +x "$work/audit"
cp "$work/audit" "$work/madaros" "$art/"
printf '%s\n' "$1" > "$art/SOURCE_COMMIT"
(cd "$art" && sha256sum audit madaros SOURCE_COMMIT > SHA256SUMS)
rec "diag-audit-sha-$(sha256sum "$art/audit" | cut -d' ' -f1)"
rec GRANITE_DIAG_BUILD_PASS
exit 0
__CA_BUNDLE__
