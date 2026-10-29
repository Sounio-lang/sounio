#!/bin/bash
#SBATCH --job-name=packet-field-oracle
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-oracle01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-oracle01/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
D="$B/sig139-oracle01"
R=/scratch/sounio-stdlib-wave3-20261027/root-integrated04
source "$R/tools-env-v2.sh"
trap 'r=$?;printf "%s\n" "$r" > "$D/WORKER_STATUS";exit "$r"' EXIT
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror "$D/reference.cpp" -o "$D/reference" > "$D/build.stdout" 2> "$D/build.stderr"
for file in "$B/sig139-packet02/results/"*.native.stdout;do
 name=${file##*/};status=0
 "$D/reference" "$file" > "$D/$name.oracle.stdout" 2> "$D/$name.oracle.stderr" ||status=$?
 expected=0
 case "$name" in *-implicit-0.seed.native.stdout|*-implicit-2.seed.native.stdout)expected=4;;esac
 printf '%s\t%s\t%s\n' "$name" "$status" "$expected" >> "$D/RESULTS.tsv"
 test "$status" = "$expected"
done
printf 'WRONG 11 22 33\n' > "$D/missing-marker"
printf 'PACKET_RETURN 11 22 33\nPACKET_RETURN 11 22 33\n' > "$D/duplicate"
printf 'PACKET_RETURN 12 22 33\n' > "$D/wrong-value"
for kind in missing-marker duplicate wrong-value;do
 status=0;"$D/reference" "$D/$kind" > "$D/$kind.stdout" 2> "$D/$kind.stderr" ||status=$?
 test "$status" != 0
 printf '%s\t%s\n' "$kind" "$status" >> "$D/CONTROLS.tsv"
done
sha256sum "$D/reference.cpp" "$D/reference" "$B/sig139-packet02/results/"*.native.stdout > "$D/INPUTS.sha256"
printf 'PACKET_FIELDS_ORACLE_AND_NEGATIVES_OK\n' > "$D/STATUS"
