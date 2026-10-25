#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:30:00
#SBATCH --export=NIL
set -u
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO168/source10
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
printf 'a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3  %s\n' "$MADAROS_RAW_BIN" | sha256sum -c - || exit 1
printf '8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc  bin/souc-lean-single-x86_64\n' | sha256sum -c - || exit 1
bash scripts/stdlib/wilcoxon_checked_gate.sh ../phase10/results > ../phase10/gate.stdout 2> ../phase10/gate.stderr
code=$?;echo "$code" > ../phase10/gate.exit
[[ "$code" == 0 ]] || exit "$code"
sha256sum ../phase10/results/gate.result ../phase10/results/compiler.sha256 > ../phase10/fresh_guard_hashes.sha256
bash scripts/stdlib/wilcoxon_checked_gate.sh ../phase10/results > ../phase10/fresh_guard.stdout 2> ../phase10/fresh_guard.stderr
code=$?;echo "$code" > ../phase10/fresh_guard.exit
[[ "$code" == 65 ]] || exit 1
grep -Fq 'fresh output directory required' ../phase10/fresh_guard.stderr || exit 1
sha256sum -c ../phase10/fresh_guard_hashes.sha256 > ../phase10/fresh_guard.verify.stdout || exit 1
echo FRESH_OUTPUT_REFUSAL_PASS > ../phase10/fresh_guard.result
