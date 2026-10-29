#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-imports-real02/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-imports-real02/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-imports-real02
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
mkdir native02
sha256sum results/*.elf reference > native02/INPUTS.before.sha256
for key in after.old after.final before.old;do
 chmod 755 "results/$key.elf"
 rc=0;timeout --kill-after=5 30 "results/$key.elf" >"native02/$key.stdout" 2>"native02/$key.stderr"||rc=$?
 echo "$rc">"native02/$key.STATUS"
 rc=0;./reference after/self-hosted/parser/ast.sio after/self-hosted/resolve/symbol.sio "native02/$key.stdout" >"native02/$key.oracle.stdout" 2>"native02/$key.oracle.stderr"||rc=$?
 echo "$rc">"native02/$key.oracle.STATUS"
done
sha256sum -c native02/INPUTS.before.sha256 > native02/INPUTS.final.verify
for f in native02/*.STATUS;do printf '%s ' "$f";cat "$f";done
cat native02/*.oracle.stdout native02/*.oracle.stderr

