#!/bin/bash
set -euo pipefail
out=/tmp/compiler-record-owner-custody-20261024
mkdir -p "$out"
copy_run(){
 local label=$1 src=$2
 mkdir -p "$out/$label"
 while IFS= read -r -d '' f;do
  local rel=${f#"$src"/}
  case "$rel" in source/*|legacy-*/*|*.elf|a.out|count_profile_calibration|*.lock|baseline.cpp)continue;;esac
  mkdir -p "$out/$label/$(dirname "$rel")"
  cp -p "$f" "$out/$label/$rel"
  cmp "$f" "$out/$label/$rel"
  printf '%s\t%s\n' "$label/$rel" "$f" >> "$out/SOURCE_MAP.tsv"
 done < <(find "$src" -maxdepth 2 -type f -size -1048576c -print0)
}
a=/scratch/sounio-stdlib-inference-scale-20261024
b=/scratch/sounio-stdlib-advanced-20261023
copy_run build-16892 "$a/compiler_ownerfix01"
copy_run witness-before-16888 "$a/record_if_owner_witness01"
copy_run fixtures-before-16897 "$a/record_if_owner_fixtures_before01"
copy_run after-16900 "$a/record_if_owner_after01"
copy_run diagnostic-current-16883 "$b/trace_context07_current"
copy_run diagnostic-materialized-16884 "$b/trace_context08_materialized"
copy_run diagnostic-borrow-16885 "$b/trace_context09_borrow"
copy_run original-16899 "$b/trace_corrected_original15"
copy_run ops-traced-16903 "$a/qualification_ops_final_traced"
copy_run summary-operational-failure-16908 "$a/qualification_summary_final02"
copy_run summary-final-16910 "$a/qualification_summary_final03"
mkdir -p "$out/witness-before-16888/witness-sources" "$out/summary-final-16910/reference-source"
for f in "$a/record_if_owner_witness01/source/"*.sio;do cp -p "$f" "$out/witness-before-16888/witness-sources/$(basename "$f").txt";done
for f in "$a/qualification_summary_final03/source/"*.cpp;do cp -p "$f" "$out/summary-final-16910/reference-source/$(basename "$f").txt";done
find "$out" -type f -size +65536c ! -name '*.gz' -exec gzip -n {} \;
cd "$out"
find . -type f ! -name COPIED.sha256 -print0 | sort -z | xargs -0 sha256sum >COPIED.sha256
sha256sum -c COPIED.sha256 > /tmp/compiler-record-owner-custody-verify.txt
