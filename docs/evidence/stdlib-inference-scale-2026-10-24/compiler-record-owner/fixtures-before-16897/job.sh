#!/bin/bash
#SBATCH --job-name=record-owner-fixtures-before
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=3072M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/record_if_owner_fixtures_before01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/record_if_owner_fixtures_before01/logs/slurm-%j.err
set -uo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
r=/scratch/sounio-stdlib-inference-scale-20261024/record_if_owner_fixtures_before01
compiler=/scratch/sounio-stdlib-inference-scale-20261024/compiler_main29ce/artifacts/madaros.elf
cd "$r"
find source -type f -print0 | sort -z | xargs -0 sha256sum >INPUTS.sha256
sha256sum "$compiler" >>INPUTS.sha256
printf 'source\tphase\texit_status\n' >RESULTS.tsv
for source in source/tests/run-pass/*.sio;do
 name=$(basename "$source" .sio)
 rc=0;"$compiler" check "$source" >"$name.check.log" 2>&1 || rc=$?
 printf '%s\tcheck\t%s\n' "$name" "$rc" >>RESULTS.tsv
 rc=0;"$compiler" build "$source" --target native -o "$name.elf" >"$name.compile.log" 2>&1 || rc=$?
 printf '%s\tcompile\t%s\n' "$name" "$rc" >>RESULTS.tsv
 [[ $rc == 0 ]] || continue
 chmod +x "$name.elf"
 rc=0;"./$name.elf" >"$name.stdout" 2>"$name.stderr" || rc=$?
 printf '%s\tnative\t%s\n' "$name" "$rc" >>RESULTS.tsv
done
rc=0;"$compiler" check source/tests/compile-fail/nested_if_incompatible_record_owner.sio >negative.check.log 2>&1 || rc=$?
printf 'incompatible-owner\tcheck-negative\t%s\n' "$rc" >>RESULTS.tsv
sha256sum -c INPUTS.sha256 >INPUTS.after.verify
find . -maxdepth 1 -type f \( -name '*.elf' -o -name '*.stdout' -o -name '*.stderr' -o -name '*.log' \) -print0 | sort -z | xargs -0 sha256sum >OUTPUTS.sha256
echo DIAGNOSTICS_CAPTURED_NOT_PASS >STATUS
