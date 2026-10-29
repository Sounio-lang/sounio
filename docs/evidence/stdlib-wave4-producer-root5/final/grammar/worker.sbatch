#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-reductions03/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-reductions03/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-final-reductions03
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
mkdir results
cp -a "$B/compiler-resumed/root5-producer-reductions02/self-hosted" .
cp -a "$B/compiler-resumed/root5-producer-reductions01/tests/fixtures/wave4_producer_root5/input" .
producer=$B/compiler-resumed/root5-final-transport04/results/generation2-producer.elf
find tests input self-hosted -type f -print0|sort -z|xargs -0 sha256sum >INPUTS.before.sha256
sha256sum "$producer">PRODUCER.before.sha256
for f in input/*.sio;do
 n=$(basename "$f" .sio);rc=0
 env SOUNIO_STDLIB_PATH="$D/self-hosted" timeout --kill-after=30 120 "$producer" "$f" "results/$n.elf" >"results/$n.stdout" 2>"results/$n.stderr"||rc=$?
 echo "$rc">"results/$n.STATUS"
 case "$n" in bare_unit|bare_unit_nested|braced_unit|valued_return|imported_bare_unit|imported_tuple_recursive_none|tuple_scalar_none|tuple_recursive_none|tuple_recursive_some) expected=0;;*)expected=1;;esac
 echo "$expected">"results/$n.EXPECTED"
 test "$rc" = "$expected"
 if [ "$expected" = 1 ];then test ! -e "results/$n.elf";fi
done
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror tests/fixtures/wave4_producer_root5/grammar/reference.cpp -o reference
env SOUNIO_STDLIB_PATH="$D/self-hosted" timeout --kill-after=30 120 "$producer" tests/fixtures/wave4_producer_root5/grammar/native.sio results/native.elf >results/native.compile.stdout 2>results/native.compile.stderr
chmod 755 results/native.elf
timeout --kill-after=5 30 results/native.elf >results/native.stdout 2>results/native.stderr
./reference results/native.stdout >results/native.oracle.stdout
sha256sum -c INPUTS.before.sha256 >INPUTS.final.verify
sha256sum -c PRODUCER.before.sha256 >PRODUCER.final.verify
echo PASS >STATUS
cat results/native.oracle.stdout

