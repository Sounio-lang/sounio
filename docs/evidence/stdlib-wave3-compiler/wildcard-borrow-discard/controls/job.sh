#!/bin/bash
#SBATCH --job-name=stdlib-discard-linear-controls
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/env-discard-controls01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/env-discard-controls01/results/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
R=$B/env-discard-controls01
D=$R/results
PAIR=$B/env-discard-checker01
ulimit -s 524288
ulimit -c 0
trap 'r=$?;echo "$r" > "$D/STATUS";exit "$r"' EXIT
sha256sum "$PAIR/results/"{before,after}.elf > "$D/TOOLS.sha256"
find "$R/inputs" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
printf 'mode\tcase\tverdict\n' > "$D/RESULTS.tsv"
for mode in before after;do
 export SOUNIO_STDLIB_PATH="$PAIR/$mode/stdlib"
 for source in "$R/inputs/"*.sio;do
  name=$(basename "$source" .sio)
  rc=0;timeout 30 "$PAIR/results/$mode.elf" "$source" > "$D/$mode.$name.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\t%s\n' "$mode" "$name" "$rc" >> "$D/RESULTS.tsv"
  if [[ $mode == before && $name != linear_discard && $name != effect_required ]];then [[ $rc == 1 ]];
  elif [[ $name == discard_then_write ]];then [[ $rc == 0 ]];
  else [[ $rc == 1 ]];! grep -q E397 "$D/$mode.$name.stdout";fi
 done
done
sha256sum -c "$D/TOOLS.sha256" > "$D/TOOLS.final.verify"
sha256sum -c "$D/INPUTS.sha256" > "$D/INPUTS.final.verify"
echo DISCARD_LINEAR_AND_LOANS_CONTROLS_OK
