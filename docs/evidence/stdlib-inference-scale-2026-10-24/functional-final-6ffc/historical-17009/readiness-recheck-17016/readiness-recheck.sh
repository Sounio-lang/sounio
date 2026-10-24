#!/bin/bash
set -uo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
base=/scratch/sounio-stdlib-inference-scale-20261024
out=$base/input_catalog_prewarm_control01/artifacts
receipt=$base/combined_final_cache02/FUNCTIONAL_GATE_RECEIPT.txt
guard=$base/nested_qualification_final02/packet/verify-ready.sh
[[ $(id -u) == 1000 ]] || exit 65
[[ ! -e "$out/readiness-recheck.status" ]] || exit 65
id -u > "$out/readiness-recheck.uid"
date -u +%FT%TZ > "$out/readiness-recheck.utc"
sha256sum "$guard" "$receipt" > "$out/readiness-recheck.INPUTS.sha256"
rc=0
bash "$guard" "$receipt" > "$out/readiness-recheck.stdout" 2> "$out/readiness-recheck.stderr" || rc=$?
printf '%s\n' "$rc" > "$out/readiness-recheck.status"
sha256sum -c "$out/readiness-recheck.INPUTS.sha256" > "$out/readiness-recheck.INPUTS.verify" || exit 1
[[ $rc == 65 ]] || exit 1
printf 'ACTUAL READINESS RECHECK REJECTED65; NO COMPILER/CAMPAIGN/SUBMISSION\n'
