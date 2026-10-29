#!/bin/bash
#SBATCH --job-name=concrete-tuple-lifetime
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --chdir=/tmp
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/concrete-tuple-lifetime83/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/concrete-tuple-lifetime83/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
T="$B/services/concrete-tuple-lifetime83"
R="$B/root-integrated04"
source "$R/tools-env-v3.sh"
export NODE_BIN="$B/services-tools-recovered/tools/node"
export SEED9="$B/compiler-resumed/generic-module-native/source/bin/souc-lean-single-x86_64"
export SEEDCC="$B/compiler-resumed/legacy-flow-bounds03/results/generation2-producer.elf"
export RAW04="$B/root-integrated04/results/madaros"
export SDK_STDLIB="$B/root-integrated05/source/stdlib"
ulimit -s 524288
ulimit -c 0
cd "$T/source"
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$T/SOURCE.before.sha256"
set +e
bash scripts/stdlib/wave4_concrete_tuple_lifetime_gate.sh "$T/results"
rc=$?
set -e
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$T/SOURCE.after.sha256"
diff -u "$T/SOURCE.before.sha256" "$T/SOURCE.after.sha256" > "$T/SOURCE.guard.diff"
printf '%s\n' "$rc" > "$T/STATUS"
exit "$rc"
