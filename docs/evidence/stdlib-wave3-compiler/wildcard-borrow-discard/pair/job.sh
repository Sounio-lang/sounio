#!/bin/bash
#SBATCH --job-name=stdlib-discard-checker-pair
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/env-discard-checker01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/env-discard-checker01/results/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
R=$B/env-discard-checker01
D=$R/results
SEED=$B/worker-portable-gate-v3/input/bin/souc-lean-single-x86_64
RAW=/scratch/sounio-stdlib-wave3-20261027/root-integrated01/results/madaros
trap 'r=$?;echo "$r" > "$D/STATUS";exit "$r"' EXIT
ulimit -s 524288
ulimit -c 0
sha256sum "$SEED" "$RAW" > "$D/TOOLS.sha256"
find "$R/inputs" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
printf 'mode\tcase\tverdict\n' > "$D/RESULTS.tsv"
for mode in before after;do
 S=$R/$mode
 mkdir -p "$S"
 tar -xf "$R/inputs/source.tar" -C "$S"
 cp "$R/inputs/$mode-check.sio" "$S/self-hosted/check/check.sio"
 cp "$R/inputs/driver.sio" "$S/driver.sio"
 export SOUNIO_STDLIB_PATH="$S/self-hosted"
 cd "$S"
 find self-hosted stdlib driver.sio -type f -print0|sort -z|xargs -0 sha256sum > "$D/$mode.SOURCE.sha256"
 timeout 240 "$SEED" driver.sio "$D/$mode.elf" > "$D/$mode.compile.stdout" 2>&1
 ! grep -Eq '^error:|^error\[|would have dropped|cannot compile an assignment' "$D/$mode.compile.stdout"
 chmod +x "$D/$mode.elf"
 for source in "$R/inputs/cases/"*.sio;do
  name=$(basename "$source" .sio)
  rc=0;timeout 30 "$D/$mode.elf" "$source" > "$D/$mode.$name.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\t%s\n' "$mode" "$name" "$rc" >> "$D/RESULTS.tsv"
  case "$name" in
   discard_effect|discard_type|borrow_conflict) [[ $rc == 1 ]];! grep -q E397 "$D/$mode.$name.stdout";;
   underscore|discard_ref|full_env_flat) if [[ $mode == before ]];then [[ $rc == 1 ]];grep -q E397 "$D/$mode.$name.stdout";else [[ $rc == 0 ]];fi;;
   *) [[ $rc == 0 ]];;
  esac
 done
 sha256sum -c "$D/$mode.SOURCE.sha256" > "$D/$mode.SOURCE.final.verify"
done
sha256sum -c "$D/TOOLS.sha256" > "$D/TOOLS.final.verify"
sha256sum -c "$D/INPUTS.sha256" > "$D/INPUTS.final.verify"
sha256sum "$D/before.elf" "$D/after.elf" > "$D/ELFS.sha256"
echo DISCARD_CHECKER_PAIR_OK_NOT_BFS_NATIVE
