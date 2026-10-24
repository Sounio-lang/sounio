#!/bin/bash
#SBATCH --job-name=record-if-owner-witness
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=3072M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/record_if_owner_witness01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/record_if_owner_witness01/logs/slurm-%j.err
set -uo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
r=/scratch/sounio-stdlib-inference-scale-20261024/record_if_owner_witness01
cd "$r"
sha256sum source/*.sio /scratch/sounio-stdlib-inference-scale-20261024/compiler/artifacts/madaros.elf /scratch/sounio-stdlib-inference-scale-20261024/compiler_main29ce/artifacts/madaros.elf > INPUTS.sha256
printf 'compiler\tvariant\tphase\texit_status\n' >RESULTS.tsv
for tool in baseline current;do
 compiler=/scratch/sounio-stdlib-inference-scale-20261024/compiler/artifacts/madaros.elf
 [[ $tool != current ]] || compiler=/scratch/sounio-stdlib-inference-scale-20261024/compiler_main29ce/artifacts/madaros.elf
 for variant in direct simple nested annotated;do
  rc=0
  "$compiler" check "source/$variant.sio" >"$tool-$variant.check.stdout" 2>"$tool-$variant.check.stderr" || rc=$?
  printf '%s\t%s\tcheck\t%s\n' "$tool" "$variant" "$rc" >>RESULTS.tsv
  rc=0
  "$compiler" build "source/$variant.sio" --target native -o "$tool-$variant.elf" >"$tool-$variant.compile.stdout" 2>"$tool-$variant.compile.stderr" || rc=$?
  printf '%s\t%s\tcompile\t%s\n' "$tool" "$variant" "$rc" >>RESULTS.tsv
  [[ $rc == 0 ]] || continue
  chmod +x "$tool-$variant.elf"
  rc=0
  /usr/bin/time -v -o "$tool-$variant.native.resource" "./$tool-$variant.elf" >"$tool-$variant.native.stdout" 2>"$tool-$variant.native.stderr" || rc=$?
  printf '%s\t%s\tnative\t%s\n' "$tool" "$variant" "$rc" >>RESULTS.tsv
 done
done
rc=0
/scratch/sounio-stdlib-inference-scale-20261024/compiler_main29ce/artifacts/madaros.elf disasm source/nested.sio > nested.disasm.stdout 2>nested.disasm.stderr || rc=$?
echo "$rc" >nested.disasm.status
sha256sum -c INPUTS.sha256 >INPUTS.after.verify
find . -maxdepth 1 -type f \( -name '*.elf' -o -name '*.stdout' -o -name '*.stderr' \) -print0 | sort -z | xargs -0 sha256sum >OUTPUTS.sha256
printf 'DIAGNOSTICS_CAPTURED_NOT_A_PASS_ASSERTION\n' >STATUS
