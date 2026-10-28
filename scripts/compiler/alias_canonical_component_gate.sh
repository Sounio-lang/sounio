#!/usr/bin/env bash
# Actual compiler-component matrix. Run under Slurm; this does not rebuild a CLI.
# A zero result qualifies explicit nominal annotations only. Open baseline
# failures remain named in STATUS and are not a generic/container acceptance.
set -euo pipefail
if test "$#" != 3;then
 echo "usage: alias_canonical_component_gate.sh COMPONENT_ELF FIXTURES FRESH_OUTPUT" >&2
 exit 64
fi
if test -z "${SLURM_JOB_ID:-}";then echo "Slurm job required" >&2;exit 64;fi
component=$(realpath "$1")
fixtures=$(realpath "$2")
out=$(realpath -m "$3")
test -x "$component"
test ! -e "$out"
mkdir -p "$out"
sha256sum "$component" > "$out/COMPONENT.sha256"
fixture_names=(aliases_fields_original chain_fields f64_identity param_return_annotated no_alias generic_shadow constructor_no_alias param_return param_return_no_alias constructor unknown cycle)
for c in "${fixture_names[@]}";do test -s "$fixtures/$c.sio";done
for c in "${fixture_names[@]}";do sha256sum "$fixtures/$c.sio";done > "$out/FIXTURES.sha256"
printf 'case\tmode\tboundary\texpected_compile\tactual_compile\texpected_native\tactual_native\n' > "$out/RESULTS.tsv"
run_case(){
 local c="$1" mode="$2" boundary="$3" expected="$4" diagnostic="$5"
 local dir="$out/$mode-$c" status=0 native=NOT_EMITTED wanted=NOT_EMITTED
 mkdir "$dir"
 /usr/bin/time -v timeout 120 "$component" "$fixtures/$c.sio" "$dir/out.elf" "$mode" > "$dir/compile.stdout" 2> "$dir/compile.stderr" || status=$?
 if test "$expected" = 0;then
  wanted=0
  test "$status" = 0
  test -s "$dir/out.elf"
  chmod +x "$dir/out.elf"
  native=0
  /usr/bin/time -v timeout 15 "$dir/out.elf" > "$dir/native.stdout" 2> "$dir/native.stderr" || native=$?
  sha256sum "$dir/out.elf" > "$dir/ELF.sha256"
 else
  test ! -e "$dir/out.elf"
  grep -F "$diagnostic" "$dir/compile.stdout" > "$dir/DIAGNOSTIC"
 fi
 printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$mode" "$boundary" "$expected" "$status" "$wanted" "$native" >> "$out/RESULTS.tsv"
 test "$status" = "$expected"
 test "$native" = "$wanted"
}
for c in aliases_fields_original chain_fields f64_identity param_return_annotated;do
 run_case "$c" before qualified_explicit_nominal_before 12 'NATIVE_REFUSAL kind=empty_stub_ud2'
 run_case "$c" after qualified_explicit_nominal_after 0 ''
done
for c in no_alias generic_shadow constructor_no_alias;do
 run_case "$c" before unchanged_positive 0 ''
 run_case "$c" after unchanged_positive 0 ''
done
# A paired no-alias witness proves this inferred-reference-return failure
# predates alias normalization. Preserve the failed compile and actual symbol;
# this row is OPEN, not an accepted language/API behavior.
for c in param_return param_return_no_alias;do
 run_case "$c" before OPEN_inferred_ref_return_receiver 12 'name=get reason=missing_lowered_body'
 run_case "$c" after OPEN_inferred_ref_return_receiver 12 'name=get reason=missing_lowered_body'
done
for mode in before after;do
 run_case constructor "$mode" checker_boundary_alias_constructor 1 'error[E015]'
 for c in unknown cycle;do
  run_case "$c" "$mode" checker_rejected_before_normalization 1 'AUTHORITATIVE_CHECK_VERDICT 1'
 done
done
sha256sum -c "$out/COMPONENT.sha256" > "$out/COMPONENT.final.verify"
sha256sum -c "$out/FIXTURES.sha256" > "$out/FIXTURES.final.verify"
printf '%s\n' ALIAS_EXPLICIT_NOMINAL_COMPONENT_ONLY_OK QUALIFIED=EXPLICIT_NOMINAL_ANNOTATIONS OPEN=INFERRED_REF_RETURN_RECEIVER CLOSED=ALIAS_CONSTRUCTOR_CHECKER_E015 INTEGRATED_RAW=NOT_VALIDATED > "$out/STATUS"
cat "$out/STATUS"
