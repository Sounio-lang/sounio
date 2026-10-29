#!/bin/bash
#SBATCH --job-name=packet-implicit-return
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-packet01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-packet01/results/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D="$B/compiler-resumed/sig139-packet01/results"
S="$B/compiler-resumed/sig139-packet01/source"
I="$B/compiler-resumed/sig139-packet01/input"
R="$B/root-integrated04"
source "$R/tools-env-v2.sh"
RAW="$B/root-integrated03/results/madaros"
SEED="$B/compiler-resumed/generic-module-native/source/bin/souc-lean-single-x86_64"
TIME=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027/tools/time
ulimit -s524288
ulimit -c0
trap 'r=$?;printf "%s\n" "$r" > "$D/WORKER_STATUS";date -u > "$D/ENDED_UTC";exit "$r"' EXIT
cd "$S"
sha256sum "$RAW" "$SEED" > "$D/COMPILERS.sha256"
find "$I" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
for source in "$I"/*.sio;do
 name=${source##*/};name=${name%.sio}
 status=0
 SOUNIO_STDLIB_PATH="$S/stdlib" timeout 60 "$RAW" check "$source" > "$D/$name.check.stdout" 2> "$D/$name.check.stderr" ||status=$?
 printf '%s\tcheck\t%s\n' "$name" "$status" >> "$D/RESULTS.tsv"
 for compiler in raw seed;do
  status=0
  if test "$compiler" = raw;then
   SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_NV2_IR_TRACE=1 "$TIME" -v timeout 60 "$RAW" compile "$source" -o "$D/$name.$compiler.elf" > "$D/$name.$compiler.compile.stdout" 2> "$D/$name.$compiler.compile.stderr" || status=$?
  else
   SOUNIO_STDLIB_PATH="$S/self-hosted" SOUNIO_NV2_IR_TRACE=1 "$TIME" -v timeout 60 "$SEED" "$source" "$D/$name.$compiler.elf" > "$D/$name.$compiler.compile.stdout" 2> "$D/$name.$compiler.compile.stderr" || status=$?
  fi
  printf '%s\t%s.compile\t%s\n' "$name" "$compiler" "$status" >> "$D/RESULTS.tsv"
  if test "$status" = 0 && test -s "$D/$name.$compiler.elf";then
   chmod +x "$D/$name.$compiler.elf"
   status=0
   timeout 5 "$D/$name.$compiler.elf" > "$D/$name.$compiler.native.stdout" 2> "$D/$name.$compiler.native.stderr" ||status=$?
   printf '%s\t%s.native\t%s\n' "$name" "$compiler" "$status" >> "$D/RESULTS.tsv"
  fi
 done
done
sha256sum -c "$D/COMPILERS.sha256" > "$D/COMPILERS.final.verify"
sha256sum -c "$D/INPUTS.sha256" > "$D/INPUTS.final.verify"
printf 'PACKET_RETURN_DIAGNOSTIC_ONLY\n' > "$D/STATUS"
