#!/bin/bash
#SBATCH --job-name=wide-mustdef-cpp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/wide-mustdef01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/wide-mustdef01/slurm-%j.err
set -euo pipefail
D=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/wide-mustdef01
R=/scratch/sounio-stdlib-wave3-20261027/root-integrated04
source "$R/tools-env-v2.sh"
trap 'r=$?;printf "%s\n" "$r" > "$D/WORKER_STATUS";exit "$r"' EXIT
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror "$D/tests/fixtures/wave4_sig139_census/reference.cpp" -o "$D/reference" > "$D/build.stdout" 2> "$D/build.stderr"
"$D/reference" "$D/source.stdout" > "$D/audit.stdout" 2> "$D/audit.stderr"
sha256sum "$D/source.stdout" "$D/reference" "$D/tests/fixtures/wave4_sig139_census/reference.cpp" "$D/tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_reference.cpp" > "$D/INPUTS.sha256"
echo CFG_AUDIT_DIAGNOSTIC_ONLY > "$D/STATUS"
