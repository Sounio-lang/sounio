#!/usr/bin/env bash
# Report existing receipts only; never fit, sample or resume the campaign.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: CAMPAIGN_ROOT FROZEN_SOURCE_ROOT FRESH_OUTPUT_ROOT' >&2;exit 64; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
campaign="$(realpath "$1")"; frozen="$(realpath "$2")"; out="$3"
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ && $(id -u) == 1000 ]] || { echo 'Slurm UID1000 execution required' >&2;exit 65; }
[[ "$out" == /* && "$out" != "$campaign"* && "$out" != "$frozen"* ]] || exit 64
mkdir "$out"
trap 's=$?;printf "%s\n" "$s" > "$out/STATUS"' EXIT
id > "$out/identity.txt"
sha256sum "$ROOT/tests/reference/count_profile_campaign_closure.cpp" "$0" > "$out/SOURCE.sha256"
# This campaign did not reach its own INPUTS-after check. Close that gap against
# its actual frozen source and the absolute compiler/functional-receipt inputs.
(cd "$frozen";sha256sum -c "$campaign/INPUTS.sha256") > "$out/CAMPAIGN_INPUTS.verify.txt"
for cell in "$campaign"/n*;do
 for batch in "$cell"/batch*;do
  [[ -f "$batch/READY" ]] || continue
  (cd "$batch";sha256sum -c FILES.sha256) >> "$out/READY_BATCH_FILES.verify.txt"
 done
done
selected_catalog() {
 (cd "$1";{
  printf '%s\0' PROTOCOL.tsv BUDGET_USED_SECONDS LAST_INVOCATION_STATUS INPUTS.sha256
  find n* -type f ! -name driver.elf -print0
 } | sort -z | xargs -0 sha256sum)
}
selected_catalog "$campaign" > "$out/ORIGINAL_OUTPUTS.before.sha256"
mkdir "$out/raw"
(cd "$campaign";{
 printf '%s\0' PROTOCOL.tsv BUDGET_USED_SECONDS LAST_INVOCATION_STATUS INPUTS.sha256
 find n* -type f ! -name driver.elf -print0
} | tar --null -T - -cf -) | tar -xf - -C "$out/raw"
selected_catalog "$out/raw" > "$out/RAW_SELECTED.sha256"
cmp "$out/ORIGINAL_OUTPUTS.before.sha256" "$out/RAW_SELECTED.sha256"
(cd "$out/raw";find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$out/RAW.sha256"
g++ --version > "$out/compiler.txt"
g++ -std=c++23 -O2 -Wall -Wextra -Werror "$ROOT/tests/reference/count_profile_campaign_closure.cpp" -o "$out/reference" > "$out/build.stdout" 2> "$out/build.stderr"
sha256sum "$out/reference" > "$out/REFERENCE.before.sha256"
"$out/reference" "$out/raw" "$out/report" > "$out/report.stdout" 2> "$out/report.stderr"
run_mutant() {
 local kind="$1" expected="$2" dir="$out/mutant-$1"
 cp -a "$out/raw" "$dir"
 case "$kind" in
  missing-cell) mv "$dir/n16_alpha0" "$dir/withheld-n16-alpha0";;
  denominator) sed -i 's/target_per_cell	128/target_per_cell	127/' "$dir/PROTOCOL.tsv";;
  seed) sed -i 's/24000000/24000001/' "$dir/n16_alpha0/batch0/datasets.tsv";;
  completion) sed -i 's/COUNT PROFILE CALIBRATION CASE COMPLETE/INCOMPLETE/' "$dir/n16_alpha0/batch0/case-0.stdout";;
  status) printf '124\n' > "$dir/n16_alpha0/batch0/case-0.status";;
  campaign-complete) touch "$dir/COMPLETE";;
 esac
 set +e
 "$out/reference" "$dir" "$out/mutant-$kind-report" > "$out/mutant-$kind.stdout" 2> "$out/mutant-$kind.stderr"
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/mutant-$kind.status"
 [[ "$status" == 2 ]]
 grep -Fqx "COUNT_PROFILE_CAMPAIGN_CLOSURE_REJECT $expected" "$out/mutant-$kind.stderr"
}
run_mutant missing-cell 'missing declared cell'
run_mutant denominator 'protocol/denominator mismatch'
run_mutant seed 'dataset seed schedule'
run_mutant completion 'native completion marker'
run_mutant status 'READY failed execution'
run_mutant campaign-complete 'incomplete campaign must not have COMPLETE marker'
# Exercise the original-output seal on an extracted copy of the interrupted
# batch, whose last case has no READY manifest. Never edit the original.
cp -a "$out/raw" "$out/mutant-original-output"
printf '%s\n' CUSTODY_MUTATION_CONTROL >> "$out/mutant-original-output/n256_alpha.25/batch32/case-6.stdout"
selected_catalog "$out/mutant-original-output" > "$out/mutant-original-output.sha256"
set +e
cmp "$out/ORIGINAL_OUTPUTS.before.sha256" "$out/mutant-original-output.sha256" > "$out/mutant-original-output.stdout" 2> "$out/mutant-original-output.stderr"
status=$?
set -e
printf '%s\n' "$status" > "$out/mutant-original-output.status"
[[ "$status" == 1 ]]
printf '%s\n' 'ORIGINAL_OUTPUT_MUTATION_CONTROL_REJECTED' > "$out/mutant-original-output.result"
selected_catalog "$campaign" > "$out/ORIGINAL_OUTPUTS.after.sha256"
cmp "$out/ORIGINAL_OUTPUTS.before.sha256" "$out/ORIGINAL_OUTPUTS.after.sha256"
sha256sum -c "$out/REFERENCE.before.sha256" > "$out/REFERENCE.verify.final.txt"
(cd "$out/raw";sha256sum -c "$out/RAW.sha256") > "$out/RAW.verify.final.txt"
sha256sum -c "$out/SOURCE.sha256" > "$out/SOURCE.verify.final.txt"
(cd "$frozen";sha256sum -c "$campaign/INPUTS.sha256") > "$out/CAMPAIGN_INPUTS.verify.final.txt"
sha256sum "$out/reference" "$out/report/"* "$out/report.stdout" "$out/RAW.sha256" "$out/SOURCE.sha256" > "$out/ARTIFACTS.sha256"
echo COUNT_PROFILE_CAMPAIGN_CLOSURE_GATE_PASS
