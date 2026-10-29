#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:20:00
#SBATCH --export=NIL
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO162/source14
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/artifacts/madaros.elf
bash scripts/stdlib/checked_distributions_gate.sh ../phase14 > ../phase14/gate.stdout 2> ../phase14/gate.stderr
echo $? > ../phase14/gate.exit
../phase14/reference --baseline ../phase02/native.stdout > ../phase14/before.decoded.stdout
sha256sum ../phase02/baseline.elf > ../phase14/before.elf.sha256
sacct -j 17095,17105 -n -P -o JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS > ../phase14/prior.sacct
