#!/bin/bash
#SBATCH --job-name=record-owner-after
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=3072M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/record_if_owner_after01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/record_if_owner_after01/logs/slurm-%j.err
set -uo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
r=/scratch/sounio-stdlib-inference-scale-20261024/record_if_owner_after01
compiler=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ownerfix01/artifacts/madaros.elf
cd "$r"
find source -type f -print0 | sort -z | xargs -0 sha256sum >INPUTS.sha256
sha256sum "$compiler" >>INPUTS.sha256
printf 'source\tphase\texit_status\n' >RESULTS.tsv
fail=0
for source in source/witness/*.sio source/fixtures/tests/run-pass/*.sio;do
 name=$(basename "$source" .sio)
 rc=0;"$compiler" check "$source" >"$name.check.log" 2>&1 || rc=$?
 printf '%s\tcheck\t%s\n' "$name" "$rc" >>RESULTS.tsv
 [[ $rc == 0 ]] || fail=1
 rc=0;"$compiler" build "$source" --target native -o "$name.elf" >"$name.compile.log" 2>&1 || rc=$?
 printf '%s\tcompile\t%s\n' "$name" "$rc" >>RESULTS.tsv
 if [[ $rc != 0 ]];then fail=1;continue;fi
 chmod +x "$name.elf"
 rc=0;"./$name.elf" >"$name.stdout" 2>"$name.stderr" || rc=$?
 printf '%s\tnative\t%s\n' "$name" "$rc" >>RESULTS.tsv
 [[ $rc == 0 ]] || fail=1
 expected='RECORD IF OWNER PASS'
 if [[ $name == nested_if_record_owner ]];then expected='NESTED IF RECORD OWNER PASS';fi
 if [[ $name == nested_if_tuple_record_owner ]];then expected='NESTED IF TUPLE RECORD OWNER PASS';fi
 rc=0;grep -Fq "$expected" "$name.stdout" || rc=$?
 printf '%s\tstdout-marker\t%s\n' "$name" "$rc" >>RESULTS.tsv
 [[ $rc == 0 ]] || fail=1
done
rc=0;"$compiler" check source/fixtures/tests/compile-fail/nested_if_incompatible_record_owner.sio >negative.check.log 2>&1 || rc=$?
raw=$rc
if [[ $rc != 0 ]] && grep -Fq 'incompatible types' negative.check.log;then rc=0;else rc=1;fi
printf 'incompatible-owner:expected-reject:actual=%s\tnegative\t%s\n' "$raw" "$rc" >>RESULTS.tsv
[[ $rc == 0 ]] || fail=1
sha256sum -c INPUTS.sha256 >INPUTS.after.verify || fail=1
find . -maxdepth 1 -type f \( -name '*.elf' -o -name '*.stdout' -o -name '*.stderr' -o -name '*.log' \) -print0 | sort -z | xargs -0 sha256sum >OUTPUTS.sha256
echo "$fail" >STATUS
exit "$fail"
