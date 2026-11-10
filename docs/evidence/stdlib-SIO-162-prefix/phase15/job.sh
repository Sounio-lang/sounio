#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:10:00
#SBATCH --export=NIL
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO162/source15
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/artifacts/madaros.elf
export SOUNIO_STDLIB_PATH=$PWD/stdlib
sha256sum scripts/ci/stdlib_exp_{series_baseline.txt,helpers_gate.sh} "$MADAROS_RAW_BIN" > ../phase15/inputs.sha256
for engine in default lean_single;do
 export SOUNIO_SOUC_ENGINE=$engine
 /usr/bin/time -v bash scripts/ci/stdlib_exp_helpers_gate.sh > ../phase15/$engine.stdout 2> ../phase15/$engine.stderr
 echo $? > ../phase15/$engine.exit
done
