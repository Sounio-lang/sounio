#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-imports-real02/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-imports-real02/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-imports-real02
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
mkdir before after results
cp -a "$B/root-integrated05/source/self-hosted" before/
cp -a "$B/root-integrated05/source/self-hosted" after/
cp tests/fixtures/wave4_producer_root5/imports/imports.before.sio before/self-hosted/resolve/imports.sio
cp tests/fixtures/wave4_producer_root5/imports/imports.after.sio after/self-hosted/resolve/imports.sio
find tests before after -type f -print0|sort -z|xargs -0 sha256sum > INPUTS.before.sha256
"$CXX" -std=c++23 -O2 tests/fixtures/wave4_producer_root5/imports/reference.cpp -o reference
for side in before after;do
 for p in old final;do
  producer="$S/bin/souc-lean-single-x86_64"
  if [ "$p" = final ];then producer="$B/compiler-resumed/legacy-flow-bounds03/results/generation2-producer.elf";fi
  key="$side.$p";rc=0
  (cd "$side";env SOUNIO_STDLIB_PATH="$D/$side/self-hosted" "$P/tools/time" -v -o "$D/results/$key.resource" timeout --kill-after=30 300 "$producer" "$D/tests/fixtures/wave4_producer_root5/imports/driver.sio" "$D/results/$key.elf") >"results/$key.stdout" 2>"results/$key.stderr"||rc=$?
  echo "$rc">"results/$key.compile.STATUS"
  if [ "$rc" = 0 ] && ! cat "results/$key.stdout" "results/$key.stderr"|grep -q 'error';then
   rc=0;timeout --kill-after=5 30 "results/$key.elf" >"results/$key.native.stdout" 2>"results/$key.native.stderr"||rc=$?
   echo "$rc">"results/$key.native.STATUS"
   rc=0;./reference after/self-hosted/parser/ast.sio after/self-hosted/resolve/symbol.sio "results/$key.native.stdout" >"results/$key.oracle.stdout" 2>"results/$key.oracle.stderr"||rc=$?
   echo "$rc">"results/$key.oracle.STATUS"
  fi
 done
done
sha256sum -c INPUTS.before.sha256 > INPUTS.final.verify
for f in results/*.STATUS;do printf '%s ' "$f";cat "$f";done
for f in results/*.stdout results/*.stderr;do perl -ne 'print "$ARGV:$_" if /error|PASS|mismatch|rows|shape/' "$f";done

