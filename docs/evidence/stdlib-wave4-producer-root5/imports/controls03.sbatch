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
mkdir controls03
cp native02/after.final.stdout controls03/good.stdout
perl -pe 'if($.==2){s/1 1 0 120/1 1 1 120/}' controls03/good.stdout >controls03/wrongkind.stdout
perl -pe 'if($.==1){s/0 0 0 0 0 -1 -1 -1 0/0 0 0 1 1 0 120 1 0/}' controls03/good.stdout >controls03/privateexport.stdout
head -55 controls03/good.stdout >controls03/missing.stdout
cat controls03/good.stdout controls03/good.stdout >controls03/duplicate.stdout
for n in good wrongkind privateexport missing duplicate;do
 rc=0;./reference after/self-hosted/parser/ast.sio after/self-hosted/resolve/symbol.sio "controls03/$n.stdout" >"controls03/$n.oracle.stdout" 2>"controls03/$n.oracle.stderr"||rc=$?
 echo "$rc">"controls03/$n.STATUS"
 if [ "$n" = good ];then test "$rc" = 0;else test "$rc" != 0;fi
done
test "$(grep -c '^error: match must be exhaustive' results/before.final.stdout)" = 1
test "$(cat results/before.final.compile.STATUS)" = 0
for p in old final;do
 ! cat "results/after.$p.stdout" "results/after.$p.stderr"|grep -q '^error'
 cmp "native02/after.$p.stdout" native02/before.old.stdout
 test "$(cat native02/after.$p.STATUS)" = 0
 test "$(cat native02/after.$p.oracle.STATUS)" = 0
done
sha256sum -c INPUTS.before.sha256 >controls03/INPUTS.final.verify
sha256sum -c native02/INPUTS.before.sha256 >controls03/NATIVE.final.verify
echo PASS >controls03/STATUS

