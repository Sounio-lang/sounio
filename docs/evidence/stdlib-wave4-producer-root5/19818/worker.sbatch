#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-option-profiles01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-option-profiles01/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-option-profiles01
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
mkdir results
cp -a "$B/compiler-resumed/root5-producer-reductions02/self-hosted" .
cp tests/fixtures/wave4_producer_root5/option/modules/*.sio self-hosted/
find tests self-hosted -type f -print0|sort -z|xargs -0 sha256sum >INPUTS.before.sha256
"$CXX" -std=c++23 -O2 -ffp-contract=off tests/fixtures/wave4_producer_root5/option/reference.cpp -o reference
for p in before after;do
 producer="$B/compiler-resumed/legacy-flow-bounds03/results/generation2-producer.elf"
 if [ "$p" = after ];then producer="$B/compiler-resumed/root5-option-join01/results/producer.elf";fi
 sha256sum "$producer" >"results/$p.PRODUCER.sha256"
 mkdir "results/$p"
 for f in tests/fixtures/wave4_producer_root5/option/input/*.sio;do
  n=$(basename "$f" .sio);rc=0
  env SOUNIO_STDLIB_PATH="$D/self-hosted" "$P/tools/time" -v -o "results/$p/$n.resource" timeout --kill-after=30 120 "$producer" "$f" "results/$p/$n.elf" >"results/$p/$n.stdout" 2>"results/$p/$n.stderr"||rc=$?
  echo "$rc">"results/$p/$n.STATUS"
 done
done
if [ "$(cat results/after/native.STATUS)" = 0 ];then
 chmod 755 results/after/native.elf
 rc=0;timeout --kill-after=5 30 results/after/native.elf >results/native.stdout 2>results/native.stderr||rc=$?
 echo "$rc">results/native.STATUS
 rc=0;./reference results/native.stdout >results/oracle.stdout 2>results/oracle.stderr||rc=$?
 echo "$rc">results/oracle.STATUS
fi
sha256sum -c INPUTS.before.sha256 >INPUTS.final.verify
for f in results/{before,after}/*.STATUS results/native.STATUS results/oracle.STATUS;do if [ -f "$f" ];then printf '%s ' "$f";cat "$f";fi;done

