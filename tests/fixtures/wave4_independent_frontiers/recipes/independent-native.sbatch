#!/bin/bash
#SBATCH --job-name=stdlib-frontiers-retained-native
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/root-independent-frontiers01/native-%j.stdout
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/root-independent-frontiers01/native-%j.stderr
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
R=$B/root-integrated05
S=$R/source
D=$B/root-independent-frontiers01
OUT=$D/native-retained01
mkdir "$OUT" "$OUT/bin"
ulimit -s 524288
ulimit -c 0
complete=0
finish(){ r=$?;trap - EXIT;set +e;(cd "$S";sha256sum -c "$R/results/SOURCE.sha256") > "$OUT/SOURCE.final.verify" 2>&1 || r=1;sha256sum -c "$OUT/INPUTS.sha256" > "$OUT/INPUTS.final.verify" 2>&1 || r=1;test "$complete" = 1 || r=1;printf '%s\n' "$r" > "$OUT/COLLECTOR_STATUS";exit "$r"; }
trap finish EXIT
(cd "$S";sha256sum -c "$R/results/SOURCE.sha256") > "$OUT/SOURCE.before.verify"
sha256sum "$P/tools/time" "$(command -v timeout)" "$D/input/native.sbatch" "$R/results/SOURCE.sha256" > "$OUT/INPUTS.sha256"
printf 'case\tmode\tprocess_status\n' > "$OUT/RESULTS.tsv"
for idx in 1 2;do
 name=graph
 if test "$idx" = 2;then name=dp54;fi
 for mode in 0 1;do
  original=$D/case-$idx/regions$mode.elf
  copy=$OUT/bin/$name.regions$mode.elf
  sha256sum "$original" >> "$OUT/INPUTS.sha256"
  cp "$original" "$copy"
  cmp "$original" "$copy"
  chmod u+x "$copy"
  sha256sum "$copy" >> "$OUT/INPUTS.sha256"
  printf '%q ' "$P/tools/time" -v -o "$OUT/$name.regions$mode.time" timeout --kill-after=30s 120 "$copy" > "$OUT/$name.regions$mode.argv"
  printf '\n' >> "$OUT/$name.regions$mode.argv"
  rc=0
  "$P/tools/time" -v -o "$OUT/$name.regions$mode.time" timeout --kill-after=30s 120 "$copy" > "$OUT/$name.regions$mode.stdout" 2> "$OUT/$name.regions$mode.stderr" || rc=$?
  printf '%s\n' "$rc" > "$OUT/$name.regions$mode.PROCESS_STATUS"
  printf '%s\t%s\t%s\n' "$name" "$mode" "$rc" >> "$OUT/RESULTS.tsv"
 done
done
complete=1
echo RETAINED_ELF_NATIVE_OBSERVATIONS_COMPLETE_NO_MARKER_ONLY_ACCEPTANCE
