#!/bin/bash
#SBATCH --job-name=norm-perf-summary
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024M
#SBATCH --time=00:03:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-advanced-20261023/normalization_perf_summary01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-advanced-20261023/normalization_perf_summary01/logs/slurm-%j.err
export PATH=/usr/bin:/bin
cd /scratch/sounio-stdlib-advanced-20261023/normalization_perf_summary01
sha256sum source/* /scratch/sounio-stdlib-advanced-20261023/normalization_perf01/PHASES.tsv > INPUTS.sha256
g++ -std=c++23 -O2 source/normalization_phase_summary.cpp -o phase_summary >compile.stdout 2>compile.stderr
rc=$?;echo "$rc" >compile.status
[[ $rc == 0 ]] || exit "$rc"
./phase_summary /scratch/sounio-stdlib-advanced-20261023/normalization_perf01/PHASES.tsv >native.stdout 2>native.stderr
rc=$?;echo "$rc" >native.status
sha256sum -c INPUTS.sha256 >INPUTS.after.verify || rc=1
sha256sum phase_summary native.stdout native.stderr >OUTPUTS.sha256
echo "$rc" >STATUS
exit "$rc"
