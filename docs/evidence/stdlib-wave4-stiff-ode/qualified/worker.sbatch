#!/bin/bash
#SBATCH --job-name=stdlib-stiff-real-mc
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/core/stiff-contract04/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/core/stiff-contract04/results/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R="$B/core/stiff-contract04"
S="$R/source"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
T="$P/tooling"
export PATH="$P/tools:$T/root/usr/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$T/root/usr/lib/x86_64-linux-gnu"
export MAGIC="$T/root/usr/lib/file/magic.mgc"
RAW="$B/root-integrated02/results/madaros"
ulimit -s524288
ulimit -c0
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
date -u > "$R/results/STARTED_UTC"
cp -a "$B/whole-integrated01/source" "$S"
cp -a "$R/input/stdlib/." "$S/stdlib/"
cp "$R/input/tests/stdlib/epistemic/test_stiff_ode_knightian_mc.sio" "$S/tests/stdlib/epistemic/"
cp -a "$R/input/tests/fixtures/wave4_stiff_ode" "$S/tests/fixtures/"
cp "$R/input/tests/reference/stiff_ode_contract_reference.cpp" "$S/tests/reference/"
cp "$R/input/scripts/stdlib/stiff_ode_contract_gate.sh" "$S/scripts/stdlib/"
cd "$S"
find stdlib -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/STDLIB.actual.sha256"
cmp "$R/input/STDLIB.sha256" "$R/results/STDLIB.actual.sha256"
sha256sum -c "$R/input/STDLIB.sha256" > "$R/results/STDLIB.before.verify"
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/INPUTS.sha256"
printf '8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932  %s\n' "$RAW" > "$R/results/RAW.sha256"
sha256sum -c "$R/results/RAW.sha256" > "$R/results/RAW.before.verify"
/usr/bin/time -v g++ -std=c++23 -O2 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror tests/reference/stiff_ode_contract_reference.cpp -o "$R/results/reference" > "$R/results/oracle-build.stdout" 2> "$R/results/oracle-build.stderr"
bash scripts/stdlib/stiff_ode_contract_gate.sh "$RAW" "$S" "$R/results/reference" "$R/results/gate" > "$R/results/gate.stdout" 2> "$R/results/gate.stderr"
sha256sum -c "$R/results/RAW.sha256" > "$R/results/RAW.final.verify"
sha256sum -c "$R/results/INPUTS.sha256" > "$R/results/INPUTS.final.verify"
sha256sum -c "$R/input/STDLIB.sha256" > "$R/results/STDLIB.final.verify"
echo STIFF_ODE_FOCUSED_REAL_MC_CPP23_OK > "$R/results/STATUS"
