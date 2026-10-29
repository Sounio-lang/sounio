#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-profiles05/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-profiles05/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-final-profiles05
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
mkdir results generated
cp -a "$B/compiler-resumed/root5-producer-reductions02/self-hosted" .
cp tests/fixtures/wave4_producer_root5/option/modules/*.sio self-hosted/
find tests self-hosted -type f -print0|sort -z|xargs -0 sha256sum > INPUTS.before.sha256
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror tests/fixtures/wave4_producer_root5/option/flat_reference.cpp -o reference
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror tests/fixtures/wave4_producer_root5/option/stress_generate.cpp -o generate
./generate generated
find generated -type f -print0|sort -z|xargs -0 sha256sum >GENERATED.before.sha256
producer="$B/compiler-resumed/root5-final-transport04/results/generation2-producer.elf"
sha256sum "$producer" >results/PRODUCER.sha256
for f in tests/fixtures/wave4_producer_root5/option/input/*.sio generated/*.sio;do
 n=$(basename "$f" .sio);rc=0
 env SOUNIO_STDLIB_PATH="$D/self-hosted" "$P/tools/time" -v -o "results/$n.resource" timeout --kill-after=30 300 "$producer" "$f" "results/$n.elf" >"results/$n.stdout" 2>"results/$n.stderr"||rc=$?
 echo "$rc">"results/$n.STATUS"
 case "$n" in
  native_flat|native_cells_array|native_transport|native_closure_scope|stale_call_none|many_functions|proof_budget_near|proof_budget_unused) expected=0;;
  *) expected=1;;
 esac
 echo "$expected">"results/$n.EXPECTED"
 test "$rc" = "$expected"
 if [ "$expected" = 1 ];then test ! -e "results/$n.elf";fi
 if [ "$expected" = 0 ];then
  ! cat "results/$n.stdout" "results/$n.stderr"|grep -q '^error'
  chmod 755 "results/$n.elf"
  rc=0;timeout --kill-after=5 30 "results/$n.elf" >"results/$n.native.stdout" 2>"results/$n.native.stderr"||rc=$?
  echo "$rc">"results/$n.native.STATUS"
  test "$rc" = 0
 fi
done
./reference results/native_flat.native.stdout >results/flat.oracle.stdout
./reference results/native_cells_array.native.stdout --cells >results/cells.oracle.stdout
printf '17 -9\n'>results/many.expected
printf '1\n'>results/near.expected
printf '17\n'>results/unused.expected
cmp results/many.expected results/many_functions.native.stdout
cmp results/near.expected results/proof_budget_near.native.stdout
cmp results/unused.expected results/proof_budget_unused.native.stdout
# A fresh valid compilation after all refusals must still publish and run.
printf 'fn main()->i32 with IO{println("RECOVERY 17");0}\n'>recovery.sio
env SOUNIO_STDLIB_PATH="$D/self-hosted" timeout --kill-after=30 120 "$producer" recovery.sio results/recovery.elf >results/recovery.stdout 2>results/recovery.stderr
chmod 755 results/recovery.elf
timeout --kill-after=5 30 results/recovery.elf >results/recovery.native.stdout
printf 'RECOVERY 17\n'>results/recovery.expected
cmp results/recovery.native.stdout results/recovery.expected
sha256sum -c INPUTS.before.sha256 >INPUTS.final.verify
sha256sum -c GENERATED.before.sha256 >GENERATED.final.verify
sha256sum -c results/PRODUCER.sha256 >PRODUCER.final.verify
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror tests/fixtures/wave4_producer_root5/option/transport_reference.cpp -o transport_reference
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror tests/fixtures/wave4_producer_root5/option/scope_reference.cpp -o scope_reference
./transport_reference results/native_transport.native.stdout >results/transport.oracle.stdout
./scope_reference results/native_closure_scope.native.stdout >results/scope.oracle.stdout
echo PASS >STATUS
cat results/*.oracle.stdout

