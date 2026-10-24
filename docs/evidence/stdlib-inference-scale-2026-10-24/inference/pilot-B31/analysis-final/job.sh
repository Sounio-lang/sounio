#!/bin/bash
#SBATCH --job-name=pilot-final-analysis
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024M
#SBATCH --time=00:03:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/benchmark_analysis_final02_scripts/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/benchmark_analysis_final02_scripts/logs/slurm-%j.err
export PATH=/usr/bin:/bin
r=/scratch/sounio-stdlib-inference-scale-20261024/profile_pilot02
s=/scratch/sounio-stdlib-inference-scale-20261024/benchmark_analysis_final02_scripts
[[ -f $r/STATUS && ! -e $r/analysis-final ]] || exit 65
mkdir "$r/analysis-final"
cd "$r/analysis-final"
cp "$s/count_profile_benchmark_reference.cpp" .
sha256sum count_profile_benchmark_reference.cpp "$r/STATUS" "$r/RUNS.tsv" "$r/REPORT.tsv" "$r/runs/"*.stdout "$r/runs/"*.resource >INPUTS.sha256
cp "$r/STATUS" ORIGINAL_PACKET_STATUS
/usr/bin/time -v -o compile.resource g++ -std=c++23 -O2 -Wall -Wextra count_profile_benchmark_reference.cpp -o analyzer >compile.stdout 2>compile.stderr
rc=$?;echo "$rc" >compile.status
[[ $rc == 0 ]] || { echo "$rc" >STATUS;exit "$rc"; }
./analyzer --pilot "$r/source-before/workload.sio" "$r/source-after/workload.sio" "$r/RUNS.tsv" /scratch/sounio-stdlib-inference-scale-20261024/compiler/artifacts/madaros.elf >REPORT.tsv 2>REPORT.stderr
rc=$?;echo "$rc" >analysis.status
sha256sum -c INPUTS.sha256 >INPUTS.after.verify || rc=1
cmp "$r/REPORT.tsv" REPORT.tsv >PRELIMINARY_REPORT.compare 2>&1
compare=$?;echo "$compare" >PRELIMINARY_REPORT.compare.status
sha256sum analyzer REPORT.tsv REPORT.stderr >OUTPUTS.sha256
echo "$rc" >STATUS
exit "$rc"
