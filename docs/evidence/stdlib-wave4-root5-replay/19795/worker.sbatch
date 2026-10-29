#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:02:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-replay-metadata01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-replay-metadata01/slurm-%j.err
set -euo pipefail
D=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-replay-metadata01
cd "$D"
find source -type f -print0 | sort -z | xargs -0 sha256sum > INPUTS.before.sha256
sha256sum root5-replay-before057.sh >> INPUTS.before.sha256
perl source/tests/fixtures/wave4_root5_replay/metadata_helpers.pl root5-replay-before057.sh source/scripts/compiler/root5_regression_replay.sh "$D"
for arm in before after; do
 rc=0
 bash source/tests/fixtures/wave4_root5_replay/metadata_controls.sh "$D/$arm.helpers.sh" "$D/$arm" >"$arm.stdout" 2>"$arm.stderr" || rc=$?
 printf '%s\n' "$rc" > "$arm.STATUS"
done
sha256sum -c INPUTS.before.sha256 > INPUTS.final.verify
test "$(cat before.STATUS)" = 1
test "$(cat after.STATUS)" = 0
