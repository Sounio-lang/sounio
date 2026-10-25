#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO168/phase08
sha256sum reference.cpp ../phase06/scale.seed.native.stdout ../phase06/scale.native.stdout > inputs.sha256
g++ -std=c++23 -O2 -Wall -Wextra reference.cpp -o reference > compile.stdout 2> compile.stderr
for engine in seed.native native;do
 set +e
 ./reference --scale "../phase06/scale.$engine.stdout" > "$engine.verify.stdout" 2> "$engine.verify.stderr"
 code=$?;echo "$code" > "$engine.verify.exit"
 set -e
done
