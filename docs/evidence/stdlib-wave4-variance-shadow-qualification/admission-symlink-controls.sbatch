#!/bin/bash
#SBATCH --job-name=shadow-admission-metadata
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:03:00
#SBATCH --export=NIL
#SBATCH --chdir=/tmp
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/variance-shadow-admission77/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/variance-shadow-admission77/slurm-%j.err
set -euo pipefail
T=/scratch/sounio-stdlib-wave3-20261027/services/variance-shadow-admission77
NODE=/scratch/sounio-stdlib-wave3-20261027/services-tools-recovered/tools/node
sha256sum "$NODE" > "$T/NODE.sha256"
cd "$T/source"
sha256sum *.mjs > "$T/SOURCE.before.sha256"
set +e
"$NODE" admission_controls.mjs "$T/results"
rc=$?
set -e
sha256sum *.mjs > "$T/SOURCE.after.sha256"
diff -u "$T/SOURCE.before.sha256" "$T/SOURCE.after.sha256" > "$T/SOURCE.guard.diff"
printf '%s\n' "$rc" > "$T/STATUS"
exit "$rc"
