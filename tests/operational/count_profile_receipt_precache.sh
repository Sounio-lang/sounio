#!/usr/bin/env bash
# Negative operational control only: never creates an accepted functional receipt.
set -euo pipefail
[[ $# == 4 && ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ && $(id -u) == 1000 ]] || exit 65
src=$(realpath "$1");compiler=$(realpath "$2");rows=$(realpath "$3");out=$4
[[ ! -e "$out" ]] || { echo 'fresh output required' >&2; exit 65; }
mkdir -p "$out";out=$(realpath "$out")
case "$out/" in "$src/"*)exit 65;;esac
(cd "$src" && find -L bin scripts self-hosted stdlib tests examples -type f -print0 | sort -z | xargs -0 sha256sum) > "$out/current-inputs.sha256"
sha256sum "$compiler" "$rows" "$src/scripts/stdlib/count_profile_calibration.sh" > "$out/INPUTS.sha256"
csha=$(sha256sum "$compiler");msha=$(sha256sum "$out/current-inputs.sha256");rsha=$(sha256sum "$rows")
cat > "$out/FORGED_PRECACHE_ROWS.receipt" <<RECEIPT
format=stdlib-advanced-functional-v1
status=0
compiler_sha256=${csha%% *}
canonical_inputs_sha256=${msha%% *}
canonical_inputs_path=$out/current-inputs.sha256
results_sha256=${rsha%% *}
results_path=$rows
RECEIPT
set +e
env COUNT_PROFILE_FUNCTIONAL_GATE_RECEIPT="$out/FORGED_PRECACHE_ROWS.receipt" MADAROS_RAW_BIN="$compiler" bash "$src/scripts/stdlib/count_profile_calibration.sh" "$src" "$out/rejected-campaign" pilot >"$out/rejection.stdout" 2>"$out/rejection.stderr"
rc=$?
set -e
printf '%s\n' "$rc" >"$out/rejection.status"
[[ $rc == 65 ]]
grep -Fq 'missing/duplicate gate result:' "$out/rejection.stderr"
[[ ! -e "$out/rejected-campaign/pilot_nested_B31/tools/compile.resource" && ! -e "$out/rejected-campaign/pilot_nested_B31/tools/count_profile_calibration" ]]
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.after.verify"
printf 'NEGATIVE CONTROL PASS: real pre-cache rows rebound to current inputs rejected65 before C++ compilation; no campaign/accepted receipt.\n' > "$out/READY"
(cd "$out" && find . -type f ! -name FILES.sha256 -print0 |sort -z|xargs -0 sha256sum >FILES.sha256)
