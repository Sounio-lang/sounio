#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:30:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO169/source14
OUT=/scratch/sounio-stdlib-inference-scale-20261024/SIO169/phase14
sha256sum -c SIO169_PHASE14.sha256 > "$OUT/inputs.verify.stdout"
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
printf 'a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3  %s\n' "$MADAROS_RAW_BIN" | sha256sum -c -
printf '8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc  bin/souc-lean-single-x86_64\n' | sha256sum -c -
set +e
bash scripts/stdlib/survival_checked_gate.sh "$OUT/results" > "$OUT/gate.stdout" 2> "$OUT/gate.stderr"
code=$?;echo "$code" > "$OUT/gate.exit"
set -e
[[ "$code" == 0 ]] || exit "$code"
sha256sum "$OUT/results/gate.result" "$OUT/results/compiler.sha256" > "$OUT/fresh_guard_hashes.sha256"
set +e
bash scripts/stdlib/survival_checked_gate.sh "$OUT/results" > "$OUT/fresh_guard.stdout" 2> "$OUT/fresh_guard.stderr"
code=$?;echo "$code" > "$OUT/fresh_guard.exit"
set -e
[[ "$code" == 65 ]]
grep -q 'fresh output directory required' "$OUT/fresh_guard.stderr"
sha256sum -c "$OUT/fresh_guard_hashes.sha256" > "$OUT/fresh_guard.verify.stdout"
echo FRESH_OUTPUT_REFUSAL_PASS > "$OUT/fresh_guard.result"
# Derived protocol view only: retain all actual numeric records, rename the
# already-verified full-family completion marker for the focused score reader.
for engine in lean_single madaros;do
 grep -qx 'SURVIVAL_ROWS_COMPLETE 5' "$OUT/results/scale.$engine.native.stdout"
 sed 's/SURVIVAL_ROWS_COMPLETE 5/RARE_GROUP_SCORE_ONLY_COMPLETE/' "$OUT/results/scale.$engine.native.stdout" > "$OUT/score_after.$engine.native.stdout"
 "$OUT/results/reference" --score "$OUT/score_after.$engine.native.stdout" > "$OUT/score_after.$engine.verify.stdout" 2> "$OUT/score_after.$engine.verify.stderr"
done
sha256sum "$OUT/results/scale."*.native.stdout "$OUT/score_after."*.native.stdout "$OUT/results/reference" > "$OUT/score_projection.sha256"
