#!/usr/bin/env bash
# Allocated native qualification; scalar/working-model controls, not calibration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: genomics_batch_gate.sh FRESH_OUTPUT_DIRECTORY}"
: "${MADAROS_RAW_BIN:?explicit source-built compiler required}"
[[ -n "${SLURM_JOB_ID:-}" && "$(id -u)" == 1000 ]] || { echo 'allocated UID1000 Slurm execution required' >&2;exit 64; }
[[ -z "${SOUNIO_SOUC_BIN:-}" ]] || { echo 'raw override bypasses engine routing' >&2;exit 64; }
mkdir -p "$(dirname "$OUT")"
if ! mkdir "$OUT";then echo "fresh output directory required: $OUT" >&2;exit 65;fi
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
ulimit -s 524288
id > "$OUT/identity.txt"
sha256sum "$MADAROS_RAW_BIN" bin/souc-lean-single-x86_64 > "$OUT/compiler.sha256"
sha256sum stdlib/genomics/{count_pipeline,count_pipeline_report,count_csr,count_identities,dynamic_normalization,dispersion_shrinkage}.sio \
 stdlib/linalg/dynamic_least_squares.sio stdlib/stats/regression/{dynamic_dispersion,dynamic_count_fisher,dynamic_count_glm}.sio \
 stdlib/stats/dynamic_multiple_testing.sio stdlib/special/checked_gamma_beta.sio \
 tests/reference/{genomics_batch_independent_reference,dispersion_shrinkage_active_reference,deep_stats_dispersion_reference}.cpp \
 tests/run-pass/stdlib_genomics_batch_pipeline{,_prior,_guards,_outlier}.sio \
 tests/run-pass/stdlib_dispersion_shrinkage_prefix.sio tests/run-pass/stdlib_deep_genomics_shrinkage.sio \
 scripts/stdlib/genomics_batch_gate.sh > "$OUT/inputs.sha256"
g++ --version > "$OUT/reference.compiler.txt"
for ref in genomics_batch_independent dispersion_shrinkage_active deep_stats_dispersion;do
 /usr/bin/time -v g++ -std=c++23 -O2 -Wall -Wextra -Werror "tests/reference/${ref}_reference.cpp" -lquadmath -o "$OUT/$ref" > "$OUT/$ref.compile.stdout" 2> "$OUT/$ref.compile.stderr"
 sha256sum "$OUT/$ref" > "$OUT/$ref.sha256"
done
no_diagnostics() {
 if grep -Eq '^error:|error\[E' "$1" "$2";then echo "compiler diagnostics: $1" >&2;return 1;fi
}
run_fixture() {
 local src="$1" name="$2" marker="$3"
 sha256sum "$src" > "$OUT/$name.source.sha256"
 for engine in lean_single madaros;do
  SOUNIO_SOUC_ENGINE="$engine" bin/souc check "$src" > "$OUT/$name.$engine.check.stdout" 2> "$OUT/$name.$engine.check.stderr"
  no_diagnostics "$OUT/$name.$engine.check.stdout" "$OUT/$name.$engine.check.stderr"
  /usr/bin/time -v env SOUNIO_SOUC_ENGINE="$engine" bin/souc compile "$src" -o "$OUT/$name.$engine.elf" > "$OUT/$name.$engine.compile.stdout" 2> "$OUT/$name.$engine.compile.stderr"
  no_diagnostics "$OUT/$name.$engine.compile.stdout" "$OUT/$name.$engine.compile.stderr"
  chmod +x "$OUT/$name.$engine.elf";sha256sum "$OUT/$name.$engine.elf" > "$OUT/$name.$engine.elf.sha256"
  set +e
  /usr/bin/time -v timeout 900 "$OUT/$name.$engine.elf" > "$OUT/$name.$engine.native.stdout" 2> "$OUT/$name.$engine.native.stderr"
  local code=$?;echo "$code" > "$OUT/$name.$engine.native.exit"
  set -e
  [[ "$code" == 0 ]]
  grep -qx "$marker" "$OUT/$name.$engine.native.stdout"
 done
 sha256sum -c "$OUT/$name.source.sha256" > "$OUT/$name.source.verify.stdout"
}
run_fixture tests/run-pass/stdlib_genomics_batch_pipeline.sio pipeline GENOMICS_BATCH_PASS
run_fixture tests/run-pass/stdlib_genomics_batch_pipeline_prior.sio prior GENOMICS_BATCH_PRIOR_PASS
run_fixture tests/run-pass/stdlib_genomics_batch_pipeline_guards.sio guards GENOMICS_BATCH_GUARDS_PASS
run_fixture tests/run-pass/stdlib_genomics_batch_pipeline_outlier.sio outlier GENOMICS_BATCH_OUTLIER_PASS
run_fixture tests/run-pass/stdlib_dispersion_shrinkage_prefix.sio prefix DISPERSION_PREFIX_PASS
run_fixture tests/run-pass/stdlib_deep_genomics_shrinkage.sio legacy_shrink 'DEEP DISPERSION PASS'
for engine in lean_single madaros;do
 "$OUT/genomics_batch_independent" "$OUT/pipeline.$engine.native.stdout" > "$OUT/pipeline.$engine.verify.stdout" 2> "$OUT/pipeline.$engine.verify.stderr"
 "$OUT/genomics_batch_independent" --prior "$OUT/prior.$engine.native.stdout" > "$OUT/prior.$engine.verify.stdout" 2> "$OUT/prior.$engine.verify.stderr"
 "$OUT/genomics_batch_independent" --outlier "$OUT/outlier.$engine.native.stdout" > "$OUT/outlier.$engine.verify.stdout" 2> "$OUT/outlier.$engine.verify.stderr"
 "$OUT/dispersion_shrinkage_active" "$OUT/prefix.$engine.native.stdout" > "$OUT/prefix.$engine.verify.stdout" 2> "$OUT/prefix.$engine.verify.stderr"
 "$OUT/deep_stats_dispersion" "$OUT/legacy_shrink.$engine.native.stdout" > "$OUT/legacy_shrink.$engine.verify.stdout" 2> "$OUT/legacy_shrink.$engine.verify.stderr"
done
# Each counterfactual edits actual native output, then requires the numerical
# parser's exact refusal code/diagnostic. Compile/import failures cannot satisfy it.
for kind in value status identity missing;do
 case "$kind" in
  value) sed '0,/^FACTOR /s/^FACTOR .*/FACTOR 2.00000000000000e+00/' "$OUT/pipeline.madaros.native.stdout" > "$OUT/mutant.$kind.stdout" ;;
  status) sed '0,/^STATUS 0 0$/s/^STATUS 0 0$/STATUS 0 99/' "$OUT/pipeline.madaros.native.stdout" > "$OUT/mutant.$kind.stdout" ;;
  identity) sed '0,/4741414141/s/4741414141/4741414142/' "$OUT/pipeline.madaros.native.stdout" > "$OUT/mutant.$kind.stdout" ;;
  missing) sed '/^GENOMICS_BATCH_PASS$/d' "$OUT/pipeline.madaros.native.stdout" > "$OUT/mutant.$kind.stdout" ;;
 esac
 cmp -s "$OUT/pipeline.madaros.native.stdout" "$OUT/mutant.$kind.stdout" && { echo "ineffective mutant $kind" >&2;exit 76; }
 set +e
 "$OUT/genomics_batch_independent" "$OUT/mutant.$kind.stdout" > "$OUT/mutant.$kind.verify.stdout" 2> "$OUT/mutant.$kind.verify.stderr"
 code=$?;echo "$code" > "$OUT/mutant.$kind.verify.exit"
 set -e
 [[ "$code" == 2 ]]
 grep -q '^GENOMICS_BATCH_ORACLE_REJECT ' "$OUT/mutant.$kind.verify.stderr"
done
sha256sum -c "$OUT/inputs.sha256" > "$OUT/source.final.verify.stdout"
printf 'GENOMICS_BATCH_GATE_PASS\n' > "$OUT/gate.result"
