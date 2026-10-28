#!/bin/bash
#SBATCH --job-name=array-cast-checker-component
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/array-cast-checker15/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/array-cast-checker15/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/services/array-cast-checker15;D=$R/results;U=$B/root-integrated01
SEED=$U/source/bin/souc-lean-single-x86_64;TIME=$B/services/bigframe-integrated01/tools/time
mkdir "$D"
trap 'r=$?;echo "$r" > "$D/STATUS";exit "$r"' EXIT
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
sha256sum "$SEED" "$TIME" > "$D/TOOLS.sha256"
ulimit -s 524288;ulimit -c 0
export SOUNIO_LOOP_REGIONS=0 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
printf 'variant\tcase\tstatus\texpected\n' > "$D/RESULTS.tsv"
mismatch=0
for variant in before after;do
 cp -a "$R/input/base" "$R/$variant"
 cp "$R/input/$variant.sio" "$R/$variant/self-hosted/check/check.sio"
 export SOUNIO_STDLIB_PATH="$R/$variant/self-hosted"
 cd "$R/$variant"
 rc=0;timeout 1800 "$TIME" -v -o "$D/$variant.build.resource" "$SEED" tests/fixtures/wave4_borrow_capacity/checker_driver.sio "$D/$variant.checker.elf" > "$D/$variant.build.stdout" 2> "$D/$variant.build.stderr" ||rc=$?
 printf '%s\tbuild\t%s\t0\n' "$variant" "$rc" >> "$D/RESULTS.tsv"
 [[ "$rc" == 0 && -s "$D/$variant.checker.elf" ]]
 if grep -Ei '^error\[E|^error:|dropped.write|unresolved' "$D/$variant.build.stdout" "$D/$variant.build.stderr" > "$D/$variant.BUILD_DIAGNOSTICS";then exit 1;fi
 chmod +x "$D/$variant.checker.elf"
 sha256sum "$D/$variant.checker.elf" >> "$D/EXECUTABLES.sha256"
 count=0
 while IFS=$'\t' read -r name expected;do
  [[ -n "$name" && -s "$R/input/controls/$name.sio" ]]
  count=$((count+1))
  want="$expected";if [[ "$variant" == before ]];then want=0;fi
  rc=0;timeout 120 "$D/$variant.checker.elf" "$R/input/controls/$name.sio" > "$D/$variant.$name.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\t%s\t%s\n' "$variant" "$name" "$rc" "$want" >> "$D/RESULTS.tsv"
  [[ "$rc" == "$want" ]] || mismatch=1
  grep -Fxq "ACTUAL_CHECKER_VERDICT $want" "$D/$variant.$name.stdout" || mismatch=1
  if [[ "$want" == 1 ]];then grep -Fq 'error[E398]' "$D/$variant.$name.stdout" || mismatch=1;fi
  if grep -E 'PARSE_REFUSED|SPECIALIZATION_REFUSED|error\[E137\]' "$D/$variant.$name.stdout" >> "$D/INVALID_REFUSALS";then mismatch=1;fi
 done < "$R/input/EXPECTED.tsv"
 [[ "$count" == 22 ]]
done
sha256sum -c "$D/INPUTS.sha256" > "$D/INPUTS.final.verify"
sha256sum -c "$D/TOOLS.sha256" > "$D/TOOLS.final.verify"
sha256sum -c "$D/EXECUTABLES.sha256" > "$D/EXECUTABLES.final.verify"
echo "CHECKER_SOURCE_COMPONENT_ONLY: no BFS/native-emission or integrated compiler acceptance; exact mode0 raw11332 native facts remain previous receipts." > "$D/SCOPE"
[[ "$mismatch" == 0 ]]
echo ARRAY_CAST_CHECKER_COMPONENT_OK
