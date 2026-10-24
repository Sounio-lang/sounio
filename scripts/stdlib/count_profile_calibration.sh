#!/usr/bin/env bash
# Bounded qualification orchestration; all numerical work is C++23 or Sounio.
# No automatic submission: invoke INSIDE an authorized Slurm allocation only.
# Nested campaigns use separate namespaces; old generated outputs remain reproducible.
# CAL_WORK is read from actual result fields, never reconstructed from RNG.
# Usage: COUNT_PROFILE_FUNCTIONAL_GATE_RECEIPT=/frozen/passing-receipt \
#        MADAROS_RAW_BIN=/attested/current/compiler.elf \
#        bash scripts/stdlib/count_profile_calibration.sh SOURCE_ROOT OUTPUT_ROOT pilot|qualification
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: SOURCE_ROOT OUTPUT_ROOT pilot|qualification' >&2; exit 64; }
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ && $(id -u) == 1000 ]] || { echo 'Slurm UID1000 execution required' >&2; exit 65; }
source_root=$(realpath "$1")
mkdir -p "$2"
output_root=$(realpath "$2")
case "$output_root/" in "$source_root/"*) echo 'output must be outside source tree' >&2; exit 65;; esac
mode=$3
wall_budget=${COUNT_PROFILE_WALL_BUDGET_SECONDS:-86400}
[[ $wall_budget =~ ^[1-9][0-9]{0,4}$ ]] && ((wall_budget<=86400)) || { echo "wall budget must be 1..86400" >&2; exit 64; }
case "$mode" in pilot) B=31; cells=('16 .25'); target=16;; qualification) B=199; cells=('16 0' '16 .25' '16 1' '64 0' '64 .25' '64 1' '256 0' '256 .25' '256 1'); target=128;; *) exit 64;; esac
[[ -n ${COUNT_PROFILE_FUNCTIONAL_GATE_RECEIPT:-} && -n ${MADAROS_RAW_BIN:-} ]] || { echo 'frozen functional receipt and attested compiler required' >&2; exit 65; }
[[ -r "$COUNT_PROFILE_FUNCTIONAL_GATE_RECEIPT" && -x "$MADAROS_RAW_BIN" ]] || exit 65
campaign="$output_root/${mode}_nested_B${B}"
mkdir -p "$campaign"/{tmp,cache,tools}
exec 9>"$campaign/campaign.lock"
flock -n 9 || { echo 'campaign already active' >&2; exit 75; }
cd "$source_root"
manifest=$(mktemp "$campaign/tmp/inputs.XXXXXX")
find -L bin scripts self-hosted stdlib tests examples -type f -print0 | sort -z | xargs -0 sha256sum > "$manifest"
# Strict, non-executable key/value receipt. Reject unknown/duplicate/empty fields.
declare -A gate=()
while IFS='=' read -r key value || [[ -n ${key:-} ]]; do
 case "$key" in format|status|compiler_sha256|canonical_inputs_sha256|canonical_inputs_path|results_sha256|results_path) ;; *) echo "unknown gate receipt field" >&2; exit 65;; esac
 [[ -n "$value" && ! -v gate[$key] ]] || { echo 'duplicate/empty gate field' >&2; exit 65; }
 gate[$key]=$value
done < "$COUNT_PROFILE_FUNCTIONAL_GATE_RECEIPT"
[[ ${#gate[@]} == 7 && ${gate[format]} == stdlib-advanced-functional-v1 && ${gate[status]} == 0 ]] || { echo 'invalid functional gate receipt' >&2; exit 65; }
for field in compiler_sha256 canonical_inputs_sha256 results_sha256; do [[ ${gate[$field]} =~ ^[0-9a-f]{64}$ ]] || exit 65; done
[[ -f ${gate[canonical_inputs_path]} && -f ${gate[results_path]} ]] || exit 65
actual=$(sha256sum "$MADAROS_RAW_BIN"); [[ ${actual%% *} == ${gate[compiler_sha256]} ]] || { echo 'gate compiler mismatch' >&2; exit 65; }
actual=$(sha256sum "${gate[canonical_inputs_path]}"); [[ ${actual%% *} == ${gate[canonical_inputs_sha256]} ]] || exit 65
cmp "$manifest" "${gate[canonical_inputs_path]}" || { echo 'gate frozen source manifest mismatch' >&2; exit 65; }
actual=$(sha256sum "${gate[results_path]}"); [[ ${actual%% *} == ${gate[results_sha256]} ]] || exit 65
results=${gate[results_path]}
awk -F '\t' 'NR==1 {if($0!="kind\tsource\texit_status")exit 1;next} NF!=3||$1==""||$2==""||$3!="0" {exit 1} END {if(NR<2)exit 1}' "$results" || { echo 'gate results malformed or nonzero' >&2; exit 65; }
require_row(){ awk -F '\t' -v k="$1" -v s="$2" '$1==k&&$2==s {count++} END {exit count!=1}' "$results" || { echo "missing/duplicate gate result: $1 $2" >&2; exit 65; }; }
# CENTRAL gate labels are exact. Never accept an old full02 receipt merely
# because somebody rebinds its source/compiler hashes to this newer tree.
for fixture in stdlib_scale_count_profile_nested_bootstrap stdlib_deep_stats_count_profile_nested stdlib_scale_count_likelihood_difference stdlib_scale_normalization_into stdlib_scale_normalization_status stdlib_scale_count_fit_trace stdlib_scale_count_profile_traced; do
 for phase in check compile native stdout-marker; do require_row "$phase" "tests/run-pass/$fixture.sio"; done
done
for stem in 'deep_stats_count_profile_bootstrap_reference:nested' 'deep_stats_count_profile_bootstrap_reference:nested-traced' count_likelihood_difference_reference count_dispersion_increment_reference normalization_into_reference; do require_row concordance "$stem"; done
for control in 'deep_stats_count_profile_bootstrap_reference:nested-fixed-alpha:expected=2:actual=2' 'deep_stats_count_profile_bootstrap_reference:nested-prefix:expected=2:actual=2' 'count_dispersion_increment_reference:orientation:expected=2:actual=2'; do require_row mutant-rejection "$control"; done

require_row input-hashes unchanged
require_row compiler-hash unchanged
sha256sum "$MADAROS_RAW_BIN" "$COUNT_PROFILE_FUNCTIONAL_GATE_RECEIPT" "${gate[results_path]}" >> "$manifest"
if [[ -f "$campaign/INPUTS.sha256" ]]; then
 cmp "$manifest" "$campaign/INPUTS.sha256" || { echo 'Source/compiler/gate identity changed; use a new campaign root' >&2; exit 65; }
else
 cp "$manifest" "$campaign/INPUTS.sha256"
 printf 'mode\t%s\nentry_policy\tnested_traced\nB\t%s\nbatch_size\t16\nserial_processes\t1\ntarget_per_cell\t%s\nmax_cumulative_wall_seconds\t%s\n' "$mode" "$B" "$target" "$wall_budget" > "$campaign/PROTOCOL.tsv"
fi
rm "$manifest"
# Accumulate wall budget across resumed invocations; this is operational timing,
# never scientific numerical verification. A timeout keeps partial raw receipts.
used=0
[[ ! -f "$campaign/BUDGET_USED_SECONDS" ]] || read -r used < "$campaign/BUDGET_USED_SECONDS"
[[ $used =~ ^[0-9]+$ && $used -lt $wall_budget ]] || { echo 'campaign wall budget exhausted' >&2; exit 75; }
started=$SECONDS
# Reserve the entire remaining budget before work. A hard kill leaves it
# exhausted; resuming then requires explicit review rather than losing elapsed time.
printf '%s\n' "$wall_budget" > "$campaign/BUDGET_USED_SECONDS"
finish(){ local rc=$?; trap - EXIT; printf '%s\n' "$((used+SECONDS-started))" > "$campaign/BUDGET_USED_SECONDS"; printf '%s\n' "$rc" > "$campaign/LAST_INVOCATION_STATUS"; exit "$rc"; }
trap finish EXIT
ulimit -s 524288
export TMPDIR="$campaign/tmp" SOUNIO_MADAROS_CACHE="$campaign/cache" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_STDLIB_PATH="$source_root/stdlib" SOUNIO_SOUC_ENGINE=madaros SOUNIO_BUILD_LOCK="$campaign/build.lock"
unset SOUNIO_SOUC_BIN SOUNIO_TEST_SOUC_BIN SOUC_NATIVE_BIN
/usr/bin/time -v -o "$campaign/tools/compile.resource" g++ -std=c++23 -O2 tests/reference/count_profile_calibration.cpp -lquadmath -o "$campaign/tools/count_profile_calibration" > "$campaign/tools/compile.stdout" 2> "$campaign/tools/compile.stderr"
sha256sum "$campaign/tools/count_profile_calibration" > "$campaign/tools/BINARY.sha256"
for ((first=0; first<target; first+=16)); do
 cell_index=0
 for cell in "${cells[@]}"; do
 read -r n alpha <<< "$cell"
 cell_dir="$campaign/n${n}_alpha${alpha}"
 mkdir -p "$cell_dir"
  batch="$cell_dir/batch${first}"
  if [[ -f "$batch/READY" ]]; then
   (cd "$batch" && sha256sum -c FILES.sha256 > resume-verify.log)
  else
  # Incomplete batches remain intact. Never overwrite or silently omit them.
  if [[ -e "$batch" ]]; then echo "incomplete batch requires explicit review: $batch" >&2; exit 75; fi
  remaining=$((wall_budget-used-SECONDS+started))
  ((remaining>0)) || { echo 'campaign wall budget exhausted' >&2; exit 75; }
  mkdir "$batch"
  read -r generation_seed bootstrap_seed < <("$campaign/tools/count_profile_calibration" --seeds "$cell_index")
  [[ $generation_seed =~ ^[0-9]+$ && $bootstrap_seed =~ ^[0-9]+$ ]] || { echo "invalid C++ seed schedule" >&2; exit 65; }
  "$campaign/tools/count_profile_calibration" --generate-nested-traced tests/run-pass/stdlib_deep_stats_count_profile_bootstrap.sio "$batch" "$n" "$alpha" "$B" 16 "$first" "$generation_seed" "$bootstrap_seed" > "$batch/generate.stdout" 2> "$batch/generate.stderr"
  set +e
  /usr/bin/time -v -o "$batch/compile.resource" timeout --kill-after=30 "$remaining" bin/souc compile "$batch/driver.sio" -o "$batch/driver.elf" > "$batch/compile.stdout" 2> "$batch/compile.stderr"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$batch/compile.status"
  ((rc==0)) || exit "$rc"
  chmod +x "$batch/driver.elf"
  sha256sum "$batch/driver.elf" > "$batch/ELF-before.sha256"
  # One ELF, independent process per observed dataset: managed temporaries do
  # not accumulate across the sixteen observations. Scientific budgets unchanged.
  for ((case_index=0; case_index<16; case_index++)); do
   remaining=$((wall_budget-used-SECONDS+started))
   ((remaining>0)) || { printf '75\n' > "$batch/native.status"; exit 75; }
   set +e
   /usr/bin/time -v -o "$batch/case-${case_index}.resource" timeout --kill-after=30 "$remaining" env COUNT_PROFILE_CALIBRATION_CASE="$case_index" "$batch/driver.elf" > "$batch/case-${case_index}.stdout" 2> "$batch/case-${case_index}.stderr"
   rc=$?
   set -e
   printf '%s\n' "$rc" > "$batch/case-${case_index}.status"
   if ((rc!=0)); then printf '%s\n' "$rc" > "$batch/native.status"; exit "$rc"; fi
   if ! grep -Fxq 'COUNT PROFILE CALIBRATION CASE COMPLETE' "$batch/case-${case_index}.stdout"; then
    printf '65\n' > "$batch/native.status"; exit 65
   fi
   sha256sum -c "$batch/ELF-before.sha256" > "$batch/case-${case_index}.elf-verify"
  done
  : > "$batch/native.stdout"
  for ((case_index=0; case_index<16; case_index++)); do cat "$batch/case-${case_index}.stdout" >> "$batch/native.stdout"; done
  printf '%s\n' 'COUNT PROFILE CALIBRATION BATCH COMPLETE' >> "$batch/native.stdout"
  printf '0\n' > "$batch/native.status"
  sha256sum -c "$batch/ELF-before.sha256" > "$batch/ELF-after.verify"
  "$campaign/tools/count_profile_calibration" --summarize-nested-traced "$B" 16 "$first" "$batch/native.stdout" > "$batch/summary.tsv" 2> "$batch/summary.stderr"
  (cd "$batch" && find . -maxdepth 1 -type f ! -name FILES.sha256 ! -name READY ! -name resume-verify.log -print0 | sort -z | xargs -0 sha256sum > FILES.sha256)
  touch "$batch/READY"
  fi
 native_files=()
 for ((completed_first=0; completed_first<=first; completed_first+=16)); do native_files+=("$cell_dir/batch${completed_first}/native.stdout"); done
 "$campaign/tools/count_profile_calibration" --summarize-nested-traced "$B" 16 0 "${native_files[@]}" > "$cell_dir/summary.tsv" 2> "$cell_dir/summary.stderr"
 cell_index=$((cell_index+1))
 done
done
sha256sum -c "$campaign/INPUTS.sha256" > "$campaign/INPUTS-after.verify"
printf '%s\n' "Protocol completed; interpret availability and conditional intervals, not universal calibration." > "$campaign/COMPLETE"
