#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
set -u
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO162/source_c21
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
printf 'a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3  %s\n' "$MADAROS_RAW_BIN" | sha256sum -c - || exit 1
sha256sum bin/souc-lean-single-x86_64 > ../ci_c21/seed.sha256
bash scripts/stdlib/checked_distributions_gate.sh ../ci_c21 > ../ci_c21/gate.stdout 2> ../ci_c21/gate.stderr
status=$?;echo "$status" > ../ci_c21/gate.exit
if grep -Eq '^error:|error\[E' ../ci_c21/seed.check.stdout ../ci_c21/seed.check.stderr;then echo 1 > ../ci_c21/diagnostic_scan.exit;exit 1;fi
echo 0 > ../ci_c21/diagnostic_scan.exit
exit "$status"
