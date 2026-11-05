#!/usr/bin/env bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --job-name=SIO586-stack-control
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/SIO586-main-a232a6b1/control-%j.out
#SBATCH --error=/scratch/SIO586-main-a232a6b1/control-%j.err
set -euo pipefail
lane=/scratch/SIO586-main-a232a6b1
out="$lane/results/stack-control"
mkdir -p "$out"
trap 'exit 143' TERM
trap 'exit 130' INT
export SOUNIO_STDLIB_PATH="$lane/source/stdlib"
ulimit -c 0
sha256sum "$lane/results/madaros" "$lane/results/stack-gate/call64.sio" > "$out/INPUTS.sha256"
printf 'stack_kib\tphase\trc\n' > "$out/matrix.tsv"
for stack in 8192 16384; do
 rc=0
 ( ulimit -S -s "$stack" && ulimit -H -s "$stack" || exit 64
   printf 'soft=%s hard=%s\n' "$(ulimit -Ss)" "$(ulimit -Hs)" > "$out/$stack.limits"
   timeout 90 "$lane/results/madaros" check "$lane/results/stack-gate/call64.sio" ) > "$out/$stack.stdout" 2> "$out/$stack.stderr" || rc=$?
 printf '%s\tcheck\t%s\n' "$stack" "$rc" >> "$out/matrix.tsv"
done
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.final.verify"
printf 'OBSERVATIONS_COMPLETE\n' > "$out/STATUS"
