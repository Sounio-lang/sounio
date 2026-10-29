#!/bin/bash
#SBATCH --job-name=root06-template-metadata
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:03:00
#SBATCH --export=NIL
#SBATCH --chdir=/tmp
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/root06-template80/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/root06-template80/slurm-%j.err
set -euo pipefail
T=/scratch/sounio-stdlib-wave3-20261027/services/root06-template80
NODE=/scratch/sounio-stdlib-wave3-20261027/services-tools-recovered/tools/node
export CAPACITY_PATCH=/scratch/sounio-stdlib-wave3-20261027/root-integrated05/capacity.patch
sha256sum "$NODE" /usr/bin/git "$CAPACITY_PATCH" > "$T/TOOLS-READONLY-INPUTS.before.sha256"
cd "$T/source"
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$T/SOURCE.before.sha256"
bash -n scripts/stdlib/root06_build_template.sh
bash -n scripts/stdlib/root06_build.sbatch.in
"$NODE" --check tests/fixtures/wave4_root06_recipe/metadata.mjs
set +e
"$NODE" tests/fixtures/wave4_root06_recipe/controls.mjs "$T/results"
rc=$?
set -e
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$T/SOURCE.after.sha256"
diff -u "$T/SOURCE.before.sha256" "$T/SOURCE.after.sha256" > "$T/SOURCE.guard.diff"
sha256sum -c "$T/TOOLS-READONLY-INPUTS.before.sha256" > "$T/TOOLS-READONLY-INPUTS.after.verify"
printf '%s\n' "$rc" > "$T/STATUS"
exit "$rc"
