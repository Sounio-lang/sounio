#!/bin/bash
#SBATCH --job-name=profile-trace-diagnostic01
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --chdir=/scratch/sounio-stdlib-advanced-20261023/trace_context07_current/source
#SBATCH --output=/scratch/sounio-stdlib-advanced-20261023/trace_context07_current/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-advanced-20261023/trace_context07_current/slurm-%j.err
set -euo pipefail
ulimit -s 524288
run=/scratch/sounio-stdlib-advanced-20261023/trace_context07_current
src=/scratch/sounio-stdlib-advanced-20261023/trace_context07_current/source
compiler=/scratch/sounio-stdlib-inference-scale-20261024/compiler_main29ce/artifacts/madaros.elf
sha256sum "$compiler" "$run/driver.sio" stdlib/stats/regression/{count_fit_trace.sio,dynamic_count_glm.sio,dynamic_dispersion.sio,dynamic_count_profile_bootstrap.sio,nb2_likelihood.sio} > "$run/INPUTS.sha256"
env -i PATH=/usr/bin:/bin SOUNIO_STDLIB_PATH="$src/stdlib" "$compiler" build "$run/driver.sio" --target native -o "$run/driver.elf" > "$run/compile.stdout" 2> "$run/compile.stderr"
chmod +x "$run/driver.elf"
sha256sum "$run/driver.elf" > "$run/ELF.sha256"
rc=0
env -i PATH=/usr/bin:/bin /usr/bin/time -v -o "$run/native.resource" "$run/driver.elf" > "$run/native.stdout" 2> "$run/native.stderr" || rc=$?
printf '%s\n' "$rc" > "$run/native.status"
grep -q TRACE_ASSERT "$run/native.stdout"
sha256sum -c "$run/INPUTS.sha256" > "$run/INPUTS.after.verify"
printf 'DIAGNOSTIC_CAPTURED\n' > "$run/STATUS"
