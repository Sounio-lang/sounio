#!/bin/bash
set -uo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
run=/scratch/sounio-stdlib-advanced-20261023/campaign_report_sized03
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-advanced-20261023/compiler/artifacts/madaros.elf
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_MADAROS_CACHE="$run/cache" TMPDIR="$run/tmp"
ulimit -s 524288
cd "$run/source" || exit 65
export SOUNIO_STDLIB_PATH="$PWD/stdlib" SOUNIO_SOUC_ENGINE=madaros
printf 'kind\tsource\texit_status\n' > "$run/artifacts/RESULTS.tsv"
fail=0
record(){ printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$run/artifacts/RESULTS.tsv"; [[ "$3" == 0 ]] || fail=1; }
rc=0
(cd "$run/raw" && sha256sum -c RAW.sha256) > "$run/artifacts/RAW.verify" 2>&1 || rc=$?
record raw-hashes same-frozen-cohort "$rc"
for stem in count_profile_campaign_report_reference deep_viz_sized_scatter_reference deep_viz_scientific_document deep_viz_dynamic_fields scientific_canvas_export; do
 rc=0
 g++ -std=c++23 -O2 -Wall -Wextra "tests/reference/$stem.cpp" -o "$run/artifacts/$stem" > "$run/artifacts/$stem.cpp.log" 2>&1 || rc=$?
 record cpp-compile "$stem" "$rc"
done
rc=0
cp examples/viz_count_profile_campaign.sio "$run/artifacts/prepared-input.sio"
"$run/artifacts/count_profile_campaign_report_reference" --emit "$run/raw" examples/viz_count_profile_campaign.sio > "$run/artifacts/derived.tsv" 2> "$run/artifacts/derive.stderr" || rc=$?
record derive original-raw "$rc"
rc=0
cmp "$run/artifacts/prepared-input.sio" examples/viz_count_profile_campaign.sio > "$run/artifacts/EMITTER.verify" 2>&1 || rc=$?
record emitter-byte-replay sized-example "$rc"
find stdlib tests scripts self-hosted examples -type f -print0 | sort -z | xargs -0 sha256sum > "$run/artifacts/INPUTS.sha256"
sha256sum "$MADAROS_RAW_BIN" > "$run/artifacts/COMPILER.sha256"
checkrun(){
 local stem=$1 src=$2 rc=0
 timeout 120 bin/souc check "$src" > "$run/artifacts/$stem.check.log" 2>&1 || rc=$?
 record check "$stem" "$rc";[[ "$rc" == 0 ]] || return
 timeout 180 bin/souc compile "$src" -o "$run/artifacts/$stem.elf" > "$run/artifacts/$stem.compile.log" 2>&1 || rc=$?
 record compile "$stem" "$rc";[[ "$rc" == 0 ]] || return
 sha256sum "$run/artifacts/$stem.elf" > "$run/artifacts/$stem.elf.sha256"
 /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$run/artifacts/$stem.resource" timeout 180 "$run/artifacts/$stem.elf" > "$run/artifacts/$stem.stdout" 2> "$run/artifacts/$stem.stderr" || rc=$?
 record native "$stem" "$rc"
}
for stem in stdlib_deep_viz_scatter_footprint stdlib_deep_viz_sized_scatter stdlib_deep_viz_scientific_document stdlib_deep_viz_count_diagnostics stdlib_deep_viz_dynamic_fields; do checkrun "$stem" "tests/run-pass/$stem.sio";done
checkrun report examples/viz_count_profile_campaign.sio
checkrun default-report examples/viz_count_profile_campaign_default.sio
rc=0
"$run/artifacts/deep_viz_sized_scatter_reference" "$run/artifacts/stdlib_deep_viz_sized_scatter.stdout" "$run/artifacts/stdlib_deep_viz_scatter_footprint.stdout" > "$run/artifacts/sized.concordance.stdout" 2> "$run/artifacts/sized.concordance.stderr" || rc=$?
record concordance sized-geometry "$rc"
for pair in 'stdlib_deep_viz_scientific_document none' 'stdlib_deep_viz_count_diagnostics --count'; do
 read -r stem mode <<< "$pair";rc=0
 if [[ "$mode" == none ]]; then "$run/artifacts/deep_viz_scientific_document" "$run/artifacts/$stem.stdout" > "$run/artifacts/$stem.concordance.stdout" 2> "$run/artifacts/$stem.concordance.stderr" || rc=$?
 else "$run/artifacts/deep_viz_scientific_document" "$mode" "$run/artifacts/$stem.stdout" > "$run/artifacts/$stem.concordance.stdout" 2> "$run/artifacts/$stem.concordance.stderr" || rc=$?;fi
 record concordance "$stem" "$rc"
done
rc=0
"$run/artifacts/deep_viz_dynamic_fields" "$run/artifacts/stdlib_deep_viz_dynamic_fields.stdout" > "$run/artifacts/dynamic-fields.concordance.stdout" 2> "$run/artifacts/dynamic-fields.concordance.stderr" || rc=$?
record concordance legacy-dynamic-fields "$rc"
rc=0
cmp /scratch/sounio-stdlib-advanced-20261023/campaign_report_native02/artifacts/native.stdout "$run/artifacts/default-report.stdout" > "$run/artifacts/DEFAULT_BYTE.verify" 2>&1 || rc=$?
record legacy-byte-replay radius1-inside "$rc"
rc=0
"$run/artifacts/count_profile_campaign_report_reference" --negative-selftest "$run/raw" "$run/artifacts/report.stdout" > "$run/artifacts/report.concordance.stdout" 2> "$run/artifacts/report.concordance.stderr" || rc=$?
record concordance sized-campaign-report "$rc"
if [[ -x "$run/artifacts/report.elf" ]]; then
 rc=0
 env SOUNIO_VIZ_CANVAS_DUMP=1 "$run/artifacts/report.elf" > "$run/artifacts/canvas.stdout" 2> "$run/artifacts/canvas.stderr" || rc=$?
 record native actual-canvas-mode "$rc"
 if [[ "$rc" == 0 ]]; then "$run/artifacts/scientific_canvas_export" "$run/artifacts/canvas.stdout" "$run/artifacts/report.ppm" "$run/artifacts/report.png" > "$run/artifacts/canvas-export.stdout" 2> "$run/artifacts/canvas-export.stderr" || rc=$?;record export actual-canvas "$rc";fi
fi
# Before witness uses only legacy API; compilation success plus native exit1
# discriminates the missing integer footprint, rather than an absent new symbol.
cd "$run/source-before" || exit 65
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
rc=0
timeout 120 bin/souc check tests/run-pass/stdlib_deep_viz_scatter_footprint.sio > "$run/artifacts/before.check.log" 2>&1 || rc=$?
record before-check legacy-witness "$rc"
if [[ "$rc" == 0 ]]; then
 timeout 180 bin/souc compile tests/run-pass/stdlib_deep_viz_scatter_footprint.sio -o "$run/artifacts/before.elf" > "$run/artifacts/before.compile.log" 2>&1 || rc=$?
 record before-compile legacy-witness "$rc"
 if [[ "$rc" == 0 ]]; then
  timeout 180 "$run/artifacts/before.elf" > "$run/artifacts/before.stdout" 2> "$run/artifacts/before.stderr" || rc=$?
  printf '%s\n' "$rc" > "$run/artifacts/before.native.status"
  [[ "$rc" == 1 ]] && record before-native expected-assertion-exit1 0 || record before-native unexpected 1
  rc=0
  "$run/artifacts/deep_viz_sized_scatter_reference" --before "$run/artifacts/before.stdout" > "$run/artifacts/before.concordance.stdout" 2> "$run/artifacts/before.concordance.stderr" || rc=$?
  record before-concordance projected-missing-footprint "$rc"
 fi
fi
cd "$run/source" || exit 65
rc=0
sha256sum -c "$run/artifacts/INPUTS.sha256" > "$run/artifacts/INPUTS.verify.txt" 2>&1 || rc=$?
record input-hashes unchanged "$rc"
rc=0
sha256sum -c "$run/artifacts/COMPILER.sha256" > "$run/artifacts/COMPILER.verify.txt" 2>&1 || rc=$?
record compiler-hash unchanged "$rc"
printf '%s\n' "$fail" > "$run/artifacts/STATUS"
exit "$fail"
