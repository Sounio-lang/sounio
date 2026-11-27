#!/usr/bin/env bash
#SBATCH --job-name=gen2-exit-group-3752
#SBATCH --partition=gpu-orangefs
#SBATCH --nodelist=gpuorangefs-multi-r740-proxmox
#SBATCH --comment=lane:llm-granite
#SBATCH --oversubscribe
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=none
#SBATCH --time=08:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/gen2-exit-group-%j.log
# Prediction registered before submission: the compiler with Linux exit_group
# reaches the fixpoint rung, gen2 and gen3 have identical full SHA256, and
# gen1/gen2 IR is identical for all six witnesses. The exit syscall changes
# native process termination after lowering, leaving their IR unchanged.
set -euo pipefail
commit=b80d8fdab1f9dc7818705178b6bf5575cb4cb86a
key=de3503ad7785d7d59e89d43e329856aec64807d2bd613240dd75715b752fa535
work=/tmp/gen2-exit-group-$SLURM_JOB_ID
art=/orangefs/training/llm-granite-n10/gen2-exit-group-$SLURM_JOB_ID
mkdir -p "$work/home" "$art"
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$1" true </dev/null; }
export HOME="$work/home" XDG_CACHE_HOME="$work/home/.cache"
export GIT_SSL_CAINFO="$work/ca.crt"
awk 'p {print} /^__CA_BUNDLE__$/ {p=1}' "$0" > "$GIT_SSL_CAINFO"
ulimit -s unlimited
git init -q "$work/src"
cd "$work/src"
git fetch -q --depth 1 --no-tags https://github.com/Sounio-lang/sounio.git "$commit"
git checkout -q --detach FETCH_HEAD
[[ $(git rev-parse HEAD) == "$commit" ]]
bash scripts/ci/gen2_receipt_gate.sh > "$art/initial-key.log" 2>&1 || true
grep -Fxq "gen2 inputs key at HEAD: $key" "$art/initial-key.log"
rec "inputs-$key"
rec "commit-$commit"
unset SOUC_BIN SOUNIO_SOUC_BIN MADAROS_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_CACHE
SOUNIO_BUILD_LOCK="$work/build.lock" bash scripts/ci/build_modular_madaros.sh "$work/gen1" > "$art/gen1-build.log" 2>&1
rec "gen1-$(sha256sum "$work/gen1" | cut -d' ' -f1)"
rc=0
srun --ntasks=1 --cpus-per-task=4 --mem=0 --gres=none --job-name=fixpoint --export=ALL \
  env MADAROS_BIN="$work/gen1" SOUNIO_MADAROS_FP_DIR="$work/fp" SOUNIO_MADAROS_FP_MIN_INTO_ACC_DONE=122 SOUNIO_MADAROS_FP_EXPECT=fixpoint \
  bash scripts/ci/madaros_fixed_point_gate.sh </dev/null > "$art/gate.log" 2>&1 || rc=$?
rec "gate-rc-$rc"
tail -n 4 "$art/gate.log" | while IFS= read -r line; do rec "gate-${line:0:100}"; done
if [[ -f "$work/fp/ir_parity.log" ]]; then cp "$work/fp/ir_parity.log" "$art/"; fi
[[ $rc == 0 ]]
grep -Eq '^reached[[:space:]]+fixpoint$' "$art/gate.log"
grep -Eq 'PASS.*6|6.*PASS' "$art/ir_parity.log"
cmp "$work/fp/madaros.gen2" "$work/fp/madaros.gen3"
for gen in gen2 gen3; do rec "$gen-$(sha256sum "$work/fp/madaros.$gen" | cut -d' ' -f1)"; done
(cd "$art" && sha256sum *.log > SHA256SUMS)
rec "IR_SIX_AND_FULL_FIXPOINT_PASS"
exit 0
__CA_BUNDLE__
