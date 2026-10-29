#!/bin/bash
#SBATCH --job-name=stdlib-total-tags-published
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/total-tags02/slurm-%j.stdout
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/total-tags02/slurm-%j.stderr
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
D=$B/compiler-resumed/total-tags02
mkdir "$D/candidate" "$D/before"
tar -xzf "$D/candidate.tar.gz" -C "$D/candidate"
mkdir -p "$D/before/self-hosted/check"
cp "$B/compiler-resumed/total-tags01/input/dependent.before.sio" "$D/before/self-hosted/check/dependent.sio"
cp "$B/compiler-resumed/total-tags01/input/epistemic.before.sio" "$D/before/self-hosted/check/epistemic.sio"
export NODE_BIN="$P/tools/node"
"$P/tools/time" -v bash "$D/candidate/scripts/compiler/total_tags_gate.sh" "$D/candidate" "$D/before" "$D/gate" "$R/results/madaros" "$R/source/bin/souc-lean-single-x86_64" "$B/root-integrated05/seed-private/generation2-producer.elf" > "$D/gate.stdout" 2> "$D/gate.stderr"
