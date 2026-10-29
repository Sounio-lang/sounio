#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-join-debug01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-join-debug01/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-join-debug01
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
mkdir results
sha256sum root5-join-debug.sio "$B/compiler-resumed/legacy-flow-bounds03/results/generation2-producer.elf" > INPUTS.before.sha256
env SOUNIO_STDLIB_PATH="$S/self-hosted" "$P/tools/time" -v -o results/producer.resource timeout --kill-after=30 300 "$B/compiler-resumed/legacy-flow-bounds03/results/generation2-producer.elf" root5-join-debug.sio results/producer.elf > results/producer.stdout 2>results/producer.stderr
chmod 755 results/producer.elf
sha256sum results/producer.elf > results/PRODUCER.sha256
for f in "$B"/compiler-resumed/root5-producer-reductions01/tests/fixtures/wave4_producer_root5/input/tuple_*.sio;do
 n=$(basename "$f" .sio);rc=0
 timeout --kill-after=30 120 results/producer.elf "$f" "results/$n.elf" >"results/$n.stdout" 2>"results/$n.stderr"||rc=$?
 echo "$rc">"results/$n.STATUS"
done
sha256sum -c INPUTS.before.sha256 > INPUTS.final.verify
for f in results/*.STATUS;do printf '%s ' "$f";cat "$f";done

