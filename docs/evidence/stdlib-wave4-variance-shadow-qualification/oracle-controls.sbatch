#!/bin/bash
#SBATCH --job-name=shadow-fixtures-cpp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --chdir=/tmp
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/variance-shadow-fixtures74/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/variance-shadow-fixtures74/slurm-%j.err
set -euo pipefail
T=/scratch/sounio-stdlib-wave3-20261027/services/variance-shadow-fixtures74
R=/scratch/sounio-stdlib-wave3-20261027/root-integrated04
source "$R/tools-env-v3.sh"
export NODE_BIN=/scratch/sounio-stdlib-wave3-20261027/services-tools-recovered/tools/node
export CXX="$R/private-tools-v3/g++"
export CXX_SHA256="$(sha256sum "$CXX" | cut -d' ' -f1)"
cd "$T/source"
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$T/SOURCE.before.sha256"
set +e
bash scripts/stdlib/wave4_variance_shadow_gate.sh --oracle-controls "$T/results"
rc=$?
set -e
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$T/SOURCE.after.sha256"
diff -u "$T/SOURCE.before.sha256" "$T/SOURCE.after.sha256" > "$T/SOURCE.guard.diff"
printf '%s\n' "$rc" > "$T/STATUS"
exit "$rc"
