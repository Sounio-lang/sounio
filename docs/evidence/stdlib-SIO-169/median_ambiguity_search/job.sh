#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024M
#SBATCH --time=00:10:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO169/median_search02
sha256sum search.cpp > inputs.sha256
g++ -std=c++23 -O2 -Wall -Wextra search.cpp -o search > compile.stdout 2> compile.stderr
./search > native.stdout 2> native.stderr
