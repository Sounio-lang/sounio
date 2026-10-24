#!/bin/bash
set -euo pipefail
r=/scratch/sounio-stdlib-advanced-20261023/calibration_final02
run=/scratch/sounio-stdlib-advanced-20261023/full02
compiler=/scratch/sounio-stdlib-advanced-20261023/compiler/artifacts/madaros.elf
baseline=/scratch/sounio-stdlib-advanced-20261023/qualification01/pilot_B31/n16_alpha.25/batch0/native.stdout
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ && $(id -u) == 1000 && $(id -g) == 1000 ]] || exit 65
[[ -f "$run/artifacts/STATUS" && $(cat "$run/artifacts/STATUS") == 0 ]] || exit 65
ulimit -s 524288
started=$SECONDS
budget=79200
finish(){ local rc=$?; trap - EXIT; printf '%s\n' "$rc" > "$r/STATUS"; printf '%s\n' "$((SECONDS-started))" > "$r/ALLOCATION_USED_SECONDS"; exit "$rc"; }
trap finish EXIT
sha256sum "$r/tools/normalize-functional-receipt.sh" "$r/tools/pilot_resource_report.cpp" "$r/dispatch.sh" > "$r/TOOLS.sha256"
# Negative operational control: full01 is not authorized despite partial successes.
rc=0
bash "$r/tools/normalize-functional-receipt.sh" /scratch/sounio-stdlib-advanced-20261023/full01 "$r/receipts/full01-must-not-exist" "$compiler" > "$r/full01-rejection.stdout" 2> "$r/full01-rejection.stderr" || rc=$?
printf '%s\n' "$rc" > "$r/full01-rejection.status"
[[ $rc == 65 && ! -e "$r/receipts/full01-must-not-exist" ]] || exit 65
bash "$r/tools/normalize-functional-receipt.sh" "$run" "$r/receipts/full02" "$compiler" > "$r/normalization.stdout" 2> "$r/normalization.stderr"
export MADAROS_RAW_BIN="$compiler" COUNT_PROFILE_FUNCTIONAL_GATE_RECEIPT="$r/receipts/full02/FUNCTIONAL.receipt"
source_root="$run/source"
# Pin the requested final calibration implementation; never use a stale package.
[[ $(sha256sum "$source_root/tests/reference/count_profile_calibration.cpp" | cut -d' ' -f1) == 4b50ff0c80b001c2d25ce13a9f2a8eea4102cd579641661957886a1a051cd118 ]] || exit 65
[[ $(sha256sum "$source_root/scripts/stdlib/count_profile_calibration.sh" | cut -d' ' -f1) == 4af9b1045abfd003399cdfff9c7126b4e80b25dc5a3bd01540f1d2d06cefe957 ]] || exit 65
remaining=$((budget-SECONDS+started)); ((remaining>0)) || exit 75
export COUNT_PROFILE_WALL_BUDGET_SECONDS="$remaining"
/usr/bin/time -v -o "$r/pilot.resource" bash "$source_root/scripts/stdlib/count_profile_calibration.sh" "$source_root" "$r/campaign" pilot > "$r/pilot.stdout" 2> "$r/pilot.stderr"
[[ -f "$r/campaign/pilot_B31/COMPLETE" ]] || exit 65
/usr/bin/time -v -o "$r/tools/resource-report-compile.resource" g++ -std=c++23 -O2 "$r/tools/pilot_resource_report.cpp" -o "$r/tools/pilot_resource_report" > "$r/tools/resource-report-compile.stdout" 2> "$r/tools/resource-report-compile.stderr"
"$r/tools/pilot_resource_report" "$r/campaign/pilot_B31/n16_alpha.25/batch0" "$baseline" > "$r/PILOT_RESOURCE_AND_PREFIX.tsv" 2> "$r/PILOT_RESOURCE_AND_PREFIX.stderr"
sha256sum "$r/tools/pilot_resource_report" "$r/PILOT_RESOURCE_AND_PREFIX.tsv" > "$r/PILOT_PROOF.sha256"
printf 'pilot native/schema/prefix/resource receipts complete\n' > "$r/PILOT_READY"
remaining=$((budget-SECONDS+started)); ((remaining>0)) || exit 75
export COUNT_PROFILE_WALL_BUDGET_SECONDS="$remaining"
printf '%s\n' "$remaining" > "$r/QUALIFICATION_WALL_BUDGET_SECONDS"
/usr/bin/time -v -o "$r/qualification.resource" bash "$source_root/scripts/stdlib/count_profile_calibration.sh" "$source_root" "$r/campaign" qualification > "$r/qualification.stdout" 2> "$r/qualification.stderr"
sha256sum -c "$r/TOOLS.sha256" > "$r/TOOLS-after.verify"
