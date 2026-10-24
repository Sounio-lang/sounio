#!/bin/bash
#SBATCH --job-name=lr-v10-fixed-cpp
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=3072M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-advanced-20261023/lr_v10_cpp01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-advanced-20261023/lr_v10_cpp01/logs/slurm-%j.err
export PATH=/usr/bin:/bin
cd /scratch/sounio-stdlib-advanced-20261023/lr_v10_cpp01
sha256sum source/* > INPUTS.sha256
g++ -std=c++23 -O2 source/count_profile_lr_v10_reference.cpp -o fixed_reference >compile.stdout 2>compile.stderr
rc=$?;echo "$rc" > compile.status
[[ $rc == 0 ]] || exit "$rc"
./fixed_reference /scratch/sounio-stdlib-advanced-20261023/lr_v10_diagnostic01/native.stdout > native.stdout 2>native.stderr
rc=$?;echo "$rc" >native.status
sha256sum fixed_reference native.stdout native.stderr > OUTPUTS.sha256
echo "$rc" >STATUS
exit "$rc"
