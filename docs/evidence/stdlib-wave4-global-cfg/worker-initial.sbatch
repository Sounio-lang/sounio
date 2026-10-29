#!/bin/bash
#SBATCH --job-name=global-cfg-cpp23
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --chdir=/tmp
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/global-cfg78/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/global-cfg78/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
T="$B/services/global-cfg78"
source "$B/root-integrated04/tools-env-v3.sh"
export CXX="$B/root-integrated04/private-tools-v3/g++"
export CXX_SHA256="$(sha256sum "$CXX" | cut -d' ' -f1)"
export WIDE_CAPTURE="$B/compiler-resumed/wide-mustdef01/source.stdout"
export SCALAR_CAPTURE="$B/root-scalar-shadow01/scalar-0.compile.stdout"
export WIDE_CAPTURE_SHA256="$(sha256sum "$WIDE_CAPTURE" | cut -d' ' -f1)"
export SCALAR_CAPTURE_SHA256="$(sha256sum "$SCALAR_CAPTURE" | cut -d' ' -f1)"
cd "$T/source"
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$T/SOURCE.before.sha256"
set +e
bash scripts/stdlib/wave4_global_cfg_gate.sh "$T/results"
rc=$?
set -e
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$T/SOURCE.after.sha256"
diff -u "$T/SOURCE.before.sha256" "$T/SOURCE.after.sha256" > "$T/SOURCE.guard.diff"
printf '%s\n' "$rc" > "$T/STATUS"
exit "$rc"
