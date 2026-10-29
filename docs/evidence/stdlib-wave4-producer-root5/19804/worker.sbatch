#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-producer-reductions02/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-producer-reductions02/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-producer-reductions02
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
mkdir self-hosted
cp -a "$S/self-hosted/." self-hosted/
cp tests/fixtures/wave4_producer_root5/modules/*.sio self-hosted/
find tests self-hosted -type f -print0 | sort -z | xargs -0 sha256sum > INPUTS.before.sha256
sha256sum "$S/bin/souc-lean-single-x86_64" "$B/compiler-resumed/legacy-flow-bounds03/results/generation2-producer.elf" "$RAW" >> INPUTS.before.sha256
for p in old final; do
 producer="$S/bin/souc-lean-single-x86_64"
 if [ "$p" = final ]; then producer="$B/compiler-resumed/legacy-flow-bounds03/results/generation2-producer.elf"; fi
 mkdir "$p"
 for f in tests/fixtures/wave4_producer_root5/input/imported_*.sio; do
  n=$(basename "$f" .sio);rc=0
  env SOUNIO_STDLIB_PATH="$D/self-hosted" "$P/tools/time" -v -o "$p/$n.resource" timeout 120 "$producer" "$f" "$p/$n.elf" >"$p/$n.stdout" 2>"$p/$n.stderr" ||rc=$?
  echo "$rc" >"$p/$n.STATUS"
 done
done
mkdir modern
for f in tests/fixtures/wave4_producer_root5/input/imported_*.sio; do
 n=$(basename "$f" .sio);rc=0
 env SOUNIO_STDLIB_PATH="$D/self-hosted" "$P/tools/time" -v -o "modern/$n.resource" timeout 120 "$RAW" check "$f" >"modern/$n.stdout" 2>"modern/$n.stderr" ||rc=$?
 echo "$rc" >"modern/$n.STATUS"
done
sha256sum -c INPUTS.before.sha256 > INPUTS.final.verify
for p in old final modern; do
 for f in "$p"/*.STATUS; do
  n=$(basename "$f" .STATUS)
  printf '%s %s ' "$p" "$n";cat "$f"
  perl -ne 'print if /error:|error\[|parse error:/' "$p/$n.stdout" "$p/$n.stderr"
 done
done

