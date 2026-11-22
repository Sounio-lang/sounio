#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:20:00
#SBATCH --export=NIL
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO162/source08
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/artifacts/madaros.elf
bash scripts/stdlib/checked_distributions_gate.sh ../phase08 > ../phase08/gate.stdout 2> ../phase08/gate.stderr
echo $? > ../phase08/gate.exit
../phase08/reference --baseline ../phase02/native.stdout > ../phase08/before.decoded.stdout
sha256sum ../phase02/baseline.elf > ../phase08/before.elf.sha256
sacct -j 17095,17105 -n -P -o JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS > ../phase08/prior.sacct
