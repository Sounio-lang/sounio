#!/bin/bash
#SBATCH --job-name=qualification-summary-final
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/qualification_summary_final02/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/qualification_summary_final02/logs/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/qualification_summary_final02
old=/scratch/sounio-stdlib-inference-scale-20261024/qualification_ops_final_traced/results
cd "$r"
trap 'echo "$?" >STATUS' EXIT
sha256sum source/* "$old/generated/traced-smoke/native.stdout" "$old/generated/traced-smoke/zero-limit.stdout" "$old/generated/traced-smoke/driver.elf" >INPUTS.sha256
g++ -std=c++23 -O2 source/count_profile_calibration.cpp -lquadmath -o count_profile_calibration >compile.stdout 2>compile.stderr
./count_profile_calibration --summarize-nested-traced 2 1 0 "$old/generated/traced-smoke/native.stdout" >traced-summary.stdout 2>traced-summary.stderr
./count_profile_calibration --summarize-nested-traced 2 1 0 "$old/generated/traced-smoke/zero-limit.stdout" >partial-summary.stdout 2>partial-summary.stderr
cells=('16 0' '16 .25' '16 1' '64 0' '64 .25' '64 1' '256 0' '256 .25' '256 1')
cell=0
for pair in "${cells[@]}";do
 read -r n alpha <<< "$pair"
 read -r generation bootstrap < <(./count_profile_calibration --seeds "$cell")
 ./count_profile_calibration --generate /scratch/sounio-stdlib-inference-scale-20261024/qualification_ops_final_traced/source/tests/run-pass/stdlib_deep_stats_count_profile_bootstrap.sio "$r/legacy-$cell" "$n" "$alpha" 31 16 0 "$generation" "$bootstrap"
 cmp "$r/legacy-$cell/driver.sio" "$old/generated/baseline-$cell/driver.sio"
 cmp "$r/legacy-$cell/datasets.tsv" "$old/generated/baseline-$cell/datasets.tsv"
 cell=$((cell+1))
done
sha256sum -c INPUTS.sha256 >INPUTS.after.verify
sha256sum count_profile_calibration traced-summary.stdout partial-summary.stdout >OUTPUTS.sha256
printf 'Final C++ summary and9-cell old generation byte identity PASS; same native stdout/ELF reused.\n' >READY
./count_profile_calibration --summarize-nested 2 1 0 "/generated/smoke/native.stdout" >nested-summary.stdout
cmp nested-summary.stdout "/nested-B2-summary.stdout"
printf "Traced scope corrected; untraced nested summary byte identical; old9-cell drivers unchanged.\n" >>READY
sha256sum source/* nested-summary.stdout >FINAL.sha256
