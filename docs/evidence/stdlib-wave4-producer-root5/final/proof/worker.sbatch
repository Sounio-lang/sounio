#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-proof-component03/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-proof-component03/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-proof-component03
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
mkdir results
find tests -type f -print0|sort -z|xargs -0 sha256sum > INPUTS.before.sha256
cp "$B/compiler-resumed/root5-final-transport04/source/self-hosted/compiler/lean_single.sio" root5-proof-component03.sio
perl -0777 -i -pe 'my $n=s/^fn main\(\)/fn private_compiler_main_unused()/m;die "main"unless$n==1;' root5-proof-component03.sio
cat tests/fixtures/wave4_producer_root5/proof/main_extended.sio.in >>root5-proof-component03.sio
sha256sum root5-proof-component03.sio >> INPUTS.before.sha256
producer="$B/compiler-resumed/root5-final-transport04/results/generation2-producer.elf"
sha256sum "$producer" >results/PRODUCER.sha256
env SOUNIO_STDLIB_PATH="$S/self-hosted" timeout --kill-after=30 300 "$producer" root5-proof-component03.sio results/component.elf >results/compile.stdout 2>results/compile.stderr
chmod 755 results/component.elf
rc=0;timeout --kill-after=10 30 results/component.elf >results/native.stdout 2>results/native.stderr||rc=$?
echo "$rc">results/native.STATUS
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror tests/fixtures/wave4_producer_root5/proof/reference_extended.cpp -o reference
rc=0;./reference results/native.stdout >results/oracle.stdout 2>results/oracle.stderr||rc=$?
echo "$rc">results/oracle.STATUS
sha256sum -c INPUTS.before.sha256 >INPUTS.final.verify
cat results/native.STATUS results/oracle.STATUS results/oracle.stdout results/oracle.stderr

