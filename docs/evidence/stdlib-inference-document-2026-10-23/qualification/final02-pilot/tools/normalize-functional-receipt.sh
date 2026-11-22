#!/bin/bash
# Operational label normalization only. Every raw row and exit status retained.
set -euo pipefail
[[ $# == 3 ]] || exit 64
run=$(realpath "$1"); dest=$2; compiler=$(realpath "$3")
[[ -f "$run/artifacts/STATUS" && $(cat "$run/artifacts/STATUS") == 0 ]] || { echo 'gate requires completed STATUS=0' >&2; exit 65; }
[[ -f "$run/artifacts/RESULTS.tsv" && -f "$run/artifacts/INPUTS.sha256" && -f "$run/artifacts/COMPILER.sha256" ]] || exit 65
awk -F '\t' 'NR==1 {if($0!="kind\tsource\texit_status")exit 1;next} NF!=3||$1==""||$2==""||$3!="0" {exit 1} END {if(NR<2)exit 1}' "$run/artifacts/RESULTS.tsv" || { echo 'full gate contains failure or malformed row' >&2; exit 65; }
[[ ! -e "$dest" ]] || { echo 'receipt destination must be fresh' >&2; exit 65; }
mkdir -p "$dest"
(cd "$run/source" && sha256sum -c "$run/artifacts/INPUTS.sha256") > "$dest/functional-inputs.verify"
sha256sum -c "$run/artifacts/COMPILER.sha256" > "$dest/functional-compiler.verify"
expected=$(awk 'NR==1{print $1}' "$run/artifacts/COMPILER.sha256")
actual=$(sha256sum "$compiler"); [[ ${actual%% *} == "$expected" ]] || exit 65
cp "$run/artifacts/RESULTS.tsv" "$dest/RESULTS.full.tsv"
printf 'row\traw_kind\traw_source\tnormalized_kind\tnormalized_source\texit_status\n' > "$dest/LABEL_MAPPING.tsv"
awk -F '\t' -v OFS='\t' -v ledger="$dest/LABEL_MAPPING.tsv" '
NR==1 {print;next}
{
 kind=$1;src=$2
 if(kind=="stdout-marker")kind="marker"
 if(kind=="concordance") {
  if(src=="deep_stats_count_profile_bootstrap_reference")src=src ":stdlib_deep_stats_count_profile_bootstrap"
  else if(src=="deep_stats_dynamic_robust_reference")src=src ":stdlib_deep_stats_dynamic_robust"
  else if(src=="deep_stats_dynamic_robust_contrast_reference")src=src ":stdlib_deep_stats_dynamic_robust_contrast"
  else if(src=="deep_count_dataset_reference")src=src ":stdlib_deep_count_dataset_reference"
  else if(src=="deep_count_dataset_reference:--workflow")src="deep_count_dataset_reference:viz_dynamic_count_workflow"
  else if(src=="deep_viz_scientific_document")src=src ":stdlib_deep_viz_scientific_document"
  else if(src=="deep_viz_scientific_document:--count")src="deep_viz_scientific_document:stdlib_deep_viz_count_diagnostics"
 }
 if(kind=="mutant-rejection"&&src=="deep_stats_count_profile_bootstrap_reference:fixed-alpha:expected=2:actual=2")src="fixed-alpha"
 print NR-1,$1,$2,kind,src,$3 >> ledger
 print kind,src,$3
}' "$dest/RESULTS.full.tsv" > "$dest/RESULTS.normalized.tsv"
# Reconstruct every original row/status from the ledger and compare losslessly.
awk -F '\t' -v OFS='\t' 'NR==1{print "kind","source","exit_status";next}{print $2,$3,$6}' "$dest/LABEL_MAPPING.tsv" > "$dest/RESULTS.reconstructed.tsv"
cmp "$dest/RESULTS.full.tsv" "$dest/RESULTS.reconstructed.tsv"
(cd "$run/source" && find -L bin scripts self-hosted stdlib tests examples -type f -print0 | sort -z | xargs -0 sha256sum) > "$dest/CANONICAL_INPUTS.sha256"
inputs=$(sha256sum "$dest/CANONICAL_INPUTS.sha256"); results=$(sha256sum "$dest/RESULTS.normalized.tsv")
printf 'format=stdlib-advanced-functional-v1\nstatus=0\ncompiler_sha256=%s\ncanonical_inputs_sha256=%s\ncanonical_inputs_path=%s\nresults_sha256=%s\nresults_path=%s\n' "$expected" "${inputs%% *}" "$dest/CANONICAL_INPUTS.sha256" "${results%% *}" "$dest/RESULTS.normalized.tsv" > "$dest/FUNCTIONAL.receipt"
sha256sum "$dest/RESULTS.full.tsv" "$dest/LABEL_MAPPING.tsv" "$dest/RESULTS.reconstructed.tsv" "$dest/RESULTS.normalized.tsv" "$dest/FUNCTIONAL.receipt" > "$dest/CUSTODY.sha256"
