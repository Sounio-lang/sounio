#!/bin/bash
#SBATCH --job-name=stdlib-oldraw-current-frontiers
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/root-independent-frontiers01/slurm-%A-%a.stdout
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/root-independent-frontiers01/slurm-%A-%a.stderr
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
RAW=$R/results/madaros
R=$B/root-integrated05
S=$R/source
D=$B/root-independent-frontiers01
OUT=$D/case-$SLURM_ARRAY_TASK_ID
mkdir "$OUT"
ulimit -s 524288
ulimit -c 0
case "$SLURM_ARRAY_TASK_ID" in
 0) relative=tests/stdlib/os/test_env_present.sio;;
 1) relative=tests/stdlib/graph/test_graph_types_e2e.sio;;
 2) relative=tests/stdlib/ode/dp54_invalid_probe.sio;;
 *) exit 2;;
esac
complete=0
finish(){
 r=$?;trap - EXIT;set +e
 (cd "$S";sha256sum -c "$R/results/SOURCE.sha256") > "$OUT/SOURCE.final.verify" 2>&1 || r=1
 "$P/tools/node" "$D/input/validate-final-modes.mjs" "$R/GIT_TREE_CENSUS.z" "$S" "$R/GIT_SOURCE_TREE" > "$OUT/MODES_LINKS.final.verify" 2>&1 || r=1
 if test -f "$OUT/INPUTS.sha256";then sha256sum -c "$OUT/INPUTS.sha256" > "$OUT/INPUTS.final.verify" 2>&1 || r=1;fi
 test "$complete" = 1 || r=1
 printf '%s\n' "$r" > "$OUT/COLLECTOR_STATUS"
 date -u > "$OUT/ENDED_UTC"
 exit "$r"
}
trap finish EXIT
printf 'a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333  %s\n' "$RAW" > "$OUT/RAW.sha256"
sha256sum -c "$OUT/RAW.sha256" > "$OUT/RAW.before.verify"
(cd "$S";sha256sum -c "$R/results/SOURCE.sha256") > "$OUT/SOURCE.before.verify"
"$P/tools/node" "$D/input/validate-final-modes.mjs" "$R/GIT_TREE_CENSUS.z" "$S" "$R/GIT_SOURCE_TREE" > "$OUT/MODES_LINKS.before.verify"
sha256sum "$RAW" "$P/tools/time" "$P/tools/node" "$(command -v timeout)" "$D/input/diagnostics.sbatch" "$D/input/validate-final-modes.mjs" "$S/$relative" "$R/SOURCE_HEAD" "$R/GIT_SOURCE_TREE" "$R/GIT_TREE_CENSUS.z" "$R/results/SOURCE.sha256" > "$OUT/INPUTS.sha256"
printf 'qualified raw04 a4f5 + current immutable R05 SDK/source237; failed R05 ELF NOT used; check/compile only; no native execution\n' > "$OUT/ADMISSION.txt"
printf '%s\n' "$relative" > "$OUT/ORIGINAL_PATH"
printf 'mode\tphase\tprocess_status\telf_exists\texplicit_diagnostics\n' > "$OUT/RESULTS.tsv"
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
for mode in 0 1;do
 for phase in check compile;do
  name=regions$mode.$phase
  args=(env "SOUNIO_LOOP_REGIONS=$mode" "$RAW" "$phase" "$S/$relative")
  if test "$phase" = compile;then args+=(-o "$OUT/regions$mode.elf");fi
  printf '%q ' "$P/tools/time" -v -o "$OUT/$name.time" timeout --kill-after=30s 1200 "${args[@]}" > "$OUT/$name.argv"
  printf '\n' >> "$OUT/$name.argv"
  rc=0
  "$P/tools/time" -v -o "$OUT/$name.time" timeout --kill-after=30s 1200 "${args[@]}" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr" || rc=$?
  printf '%s\n' "$rc" > "$OUT/$name.PROCESS_STATUS"
  elf=no;diagnostics=no
  if test -f "$OUT/regions$mode.elf";then elf=yes;sha256sum "$OUT/regions$mode.elf" >> "$OUT/ELFS.sha256";fi
  if grep -Ei 'error\[E|error:|parse error|AST closure incomplete|refus|IR_INVALID_REG|field initializer mismatch' "$OUT/$name.stdout" "$OUT/$name.stderr" > "$OUT/$name.DIAGNOSTICS";then diagnostics=yes;fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$phase" "$rc" "$elf" "$diagnostics" >> "$OUT/RESULTS.tsv"
 done
done
complete=1
echo OBSERVATIONAL_COMPILER_FRONTIERS_CAPTURED_NOT_TEST_PASS
