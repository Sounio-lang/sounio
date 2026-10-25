#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
set -eu
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO167/phase10
sha256sum reference.cpp ../phase08/nct.native.stdout > inputs.sha256
g++ -std=c++23 -O2 reference.cpp -o reference
./reference ../phase08/nct.native.stdout > verify.stdout 2> verify.stderr
echo POSITIVE_REFERENCE_PASS > gate.result
