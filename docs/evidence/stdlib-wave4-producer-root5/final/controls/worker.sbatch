#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-controls02/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-controls02/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-final-controls02
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
mkdir results
F=$B/compiler-resumed/root5-final-profiles05
Q=$B/compiler-resumed/root5-proof-component03
producer=$B/compiler-resumed/root5-final-transport04/results/generation2-producer.elf
sha256sum "$producer" "$F/reference" "$Q/reference" "$Q/results/native.stdout" "$F/results/native_flat.native.stdout" "$F/results/native_cells_array.native.stdout" >INPUTS.before.sha256
find tests -type f -print0|sort -z|xargs -0 sha256sum >>INPUTS.before.sha256
perl tests/fixtures/wave4_producer_root5/proof/source_guard.pl "$B/compiler-resumed/legacy-flow-bounds03/source/self-hosted/compiler/lean_single.sio" "$B/compiler-resumed/root5-final-transport04/source/self-hosted/compiler/lean_single.sio" >results/source.guard
env SOUNIO_STDLIB_PATH="$S/self-hosted" timeout --kill-after=10 120 "$producer" tests/fixtures/wave4_producer_root5/option/input/native_closure_scope.sio results/scope.elf >results/compile.stdout 2>results/compile.stderr
chmod 755 results/scope.elf
timeout --kill-after=5 30 results/scope.elf >results/scope.stdout 2>results/scope.stderr
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror tests/fixtures/wave4_producer_root5/option/scope_reference.cpp -o scope_reference
./scope_reference results/scope.stdout >results/scope.oracle.stdout
cp "$F/results/native_flat.native.stdout" results/flat.good
perl -pe 'if($.==5){s/^(4) [0-9]+ /$1 9221120237041090560 /}' results/flat.good >results/flat.nan
perl -pe 'if($.==1){s/37/38/}' results/flat.good >results/flat.wrong
head -12 results/flat.good >results/flat.missing
cat results/flat.good results/flat.good >results/flat.duplicate
for n in nan wrong missing duplicate;do
 rc=0;"$F/reference" "results/flat.$n" >"results/$n.oracle.stdout" 2>"results/$n.oracle.stderr"||rc=$?
 echo "$rc">"results/$n.STATUS";test "$rc" != 0
done
perl -pe 'if($.==2){s/0 0/0 1/}' "$F/results/native_cells_array.native.stdout" >results/cells.badpayload
rc=0;"$F/reference" results/cells.badpayload --cells >results/cells.oracle.stdout 2>results/cells.oracle.stderr||rc=$?
echo "$rc">results/cells.STATUS;test "$rc" != 0
perl -pe 'if($.==1){s/24 72/24 73/}' "$Q/results/native.stdout" >results/proof.badopcode
rc=0;"$Q/reference" results/proof.badopcode >results/proof.oracle.stdout 2>results/proof.oracle.stderr||rc=$?
echo "$rc">results/proof.STATUS;test "$rc" != 0
sha256sum -c INPUTS.before.sha256 >INPUTS.final.verify
perl -pe 'if($.==1){s/17/18/}' "$F/results/native_transport.native.stdout" >results/transport.badvalue
rc=0;"$F/transport_reference" results/transport.badvalue >results/transport.oracle.stdout 2>results/transport.oracle.stderr||rc=$?
echo "$rc">results/transport.STATUS;test "$rc" != 0
echo PASS>STATUS
cat results/source.guard results/scope.oracle.stdout

