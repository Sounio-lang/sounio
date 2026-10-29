#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-exact-consumers02/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-exact-consumers02/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-exact-consumers02
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
mkdir results
cp -a "$B/root-integrated05/source/self-hosted" .
sha256sum "$B/root-integrated05/SOURCE_HEAD" >INPUTS.before.sha256
find tests self-hosted -type f -print0|sort -z|xargs -0 sha256sum >>INPUTS.before.sha256
perl tests/fixtures/wave4_producer_root5/exact/generate.pl "$B/root-integrated05/source" tests/fixtures/wave4_producer_root5/exact "$D" >results/generate.stdout 2>results/generate.stderr
sha256sum driver.sio EXTRACTIONS.json >GENERATED.before.sha256
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror tests/fixtures/wave4_producer_root5/exact/reference.cpp -o reference
for p in before after;do
 producer="$B/compiler-resumed/legacy-flow-bounds03/results/generation2-producer.elf"
 if [ "$p" = after ];then producer="$B/compiler-resumed/root5-final-transport04/results/generation2-producer.elf";fi
 sha256sum "$producer" >"results/$p.PRODUCER.sha256"
 rc=0;env SOUNIO_STDLIB_PATH="$D/self-hosted" "$P/tools/time" -v -o "results/$p.resource" timeout --kill-after=30 300 "$producer" driver.sio "results/$p.elf" >"results/$p.stdout" 2>"results/$p.stderr"||rc=$?
 echo "$rc">"results/$p.STATUS"
done
if [ "$(cat results/after.STATUS)" = 0 ];then
 chmod 755 results/after.elf
 rc=0;timeout --kill-after=5 30 results/after.elf >results/native.stdout 2>results/native.stderr||rc=$?
 echo "$rc">results/native.STATUS
 rc=0;./reference self-hosted/parser/ast.sio self-hosted/check/types.sio results/native.stdout >results/oracle.stdout 2>results/oracle.stderr||rc=$?
 echo "$rc">results/oracle.STATUS
fi
sha256sum -c INPUTS.before.sha256 >INPUTS.final.verify
sha256sum -c GENERATED.before.sha256 >GENERATED.final.verify
for f in results/*.STATUS;do printf '%s ' "$f";cat "$f";done
cat results/oracle.stdout results/oracle.stderr 2>/dev/null ||true

