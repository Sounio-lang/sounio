#!/usr/bin/env bash
# Stage-bound schema-2 formatter component; no compiler bootstrap.
set -euo pipefail
test "$#" = 4 || { echo 'usage: nv2_trace_full_gate.sh COMPONENT ORACLE SCHEMA1_LOG FRESH_OUTPUT' >&2;exit 64; }
test -n "${SLURM_JOB_ID:-}" || { echo 'Slurm required' >&2;exit 64; }
component=$(realpath "$1");oracle=$(realpath "$2");schema1=$(realpath "$3");out=$(realpath -m "$4")
test -x "$component";test -x "$oracle";test -s "$schema1";test ! -e "$out";mkdir -p "$out"
sha256sum "$component" "$oracle" "$schema1" > "$out/ARTIFACTS.sha256"
printf 'phase\texpected\tactual\n' > "$out/RESULTS.tsv"
TIME_BIN=${TIME_BIN:-/usr/bin/time}
bad=0;last=0
phase(){ local name="$1" expected="$2";shift 2;last=0;"$TIME_BIN" -v "$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || last=$?;printf '%s\t%s\t%s\n' "$name" "$expected" "$last" >> "$out/RESULTS.tsv";if test "$last" != "$expected";then bad=1;fi; }
for stage in source prepared;do
 phase "$stage-valid" 0 env SOUNIO_NV2_IR_TRACE_FULL=1 SOUNIO_NV2_IR_TRACE_FULL_STAGE="$stage" timeout 30 "$component" valid
 phase "$stage-oracle" 0 "$oracle" --probe "$stage" "$out/$stage-valid.stdout"
 other=prepared;if test "$stage" = prepared;then other=source;fi
 phase "$stage-reject-other" 1 "$oracle" --probe "$other" "$out/$stage-valid.stdout"
 phase "$stage-legacy" 0 env -u SOUNIO_NV2_IR_TRACE_FULL SOUNIO_NV2_IR_TRACE_FULL_STAGE="$stage" timeout 30 "$component" valid
 printf 'LEGACY_CONTROL_BYTES\n' > "$out/legacy.expected"
 cmp "$out/legacy.expected" "$out/$stage-legacy.stdout" > "$out/$stage-legacy.verify" || bad=1
 for control in truncate duplicate total args bits name float-word unknown end stage;do
  phase "$stage-mutate-$control" 0 "$oracle" --mutate "$control" "$out/$stage-valid.stdout" "$out/$stage-control-$control.stdout"
  phase "$stage-reject-$control" 1 "$oracle" --probe "$stage" "$out/$stage-control-$control.stdout"
 done
 for profile in extern0 extern1 extern2 global_base global_large global_maxextent global_raw;do
  phase "$stage-$profile" 0 env SOUNIO_NV2_IR_TRACE_FULL=1 SOUNIO_NV2_IR_TRACE_FULL_STAGE="$stage" timeout 30 "$component" "$profile"
  phase "$stage-profile-$profile" 0 "$oracle" --profile "$profile" "$stage" "$out/$stage-$profile.stdout"
 done
 for control in data-negative-offset data-negative-size data-overflow data-body data-not-global inactive-vreg;do
  phase "$stage-data-mutate-$control" 0 "$oracle" --mutate "$control" "$out/$stage-global_large.stdout" "$out/$stage-data-control-$control.stdout"
  phase "$stage-data-reject-$control" 1 "$oracle" --check "$stage" "$out/$stage-data-control-$control.stdout"
 done
 for control in stub-label stub-frame inactive-vreg;do
  phase "$stage-extern-mutate-$control" 0 "$oracle" --mutate "$control" "$out/$stage-extern2.stdout" "$out/$stage-extern-control-$control.stdout"
  phase "$stage-extern-reject-$control" 1 "$oracle" --check "$stage" "$out/$stage-extern-control-$control.stdout"
 done
 phase "$stage-defined-vreg-mutate" 0 "$oracle" --mutate defined-vreg "$out/$stage-valid.stdout" "$out/$stage-defined-vreg.stdout"
 phase "$stage-defined-vreg-reject" 1 "$oracle" --check "$stage" "$out/$stage-defined-vreg.stdout"
 phase "$stage-data-truncate" 0 "$oracle" --mutate truncate "$out/$stage-global_large.stdout" "$out/$stage-data-partial.stdout"
 phase "$stage-data-reject-partial" 1 "$oracle" --check "$stage" "$out/$stage-data-partial.stdout"
 for mode in stub_badreg stub_badlabels stub_badpadding defined_badreg global_negative_offset global_negative_size global_overflow global_body global_badreg global_badlabels;do
  phase "$stage-$mode" 0 env SOUNIO_NV2_IR_TRACE_FULL=1 SOUNIO_NV2_IR_TRACE_FULL_STAGE="$stage" timeout 30 "$component" "$mode"
  phase "$stage-refusal-$mode" 0 "$oracle" --refusal "$stage" 4 "$out/$stage-$mode.stdout"
  phase "$stage-reject-$mode" 1 "$oracle" --check "$stage" "$out/$stage-$mode.stdout"
 done
 for pair in badargs:8 badname:7 badregion:6 badparam:4 budget:5;do
  mode="${pair%:*}";reason="${pair#*:}"
  phase "$stage-$mode" 0 env SOUNIO_NV2_IR_TRACE_FULL=1 SOUNIO_NV2_IR_TRACE_FULL_STAGE="$stage" timeout 30 "$component" "$mode"
  phase "$stage-refusal-$mode" 0 "$oracle" --refusal "$stage" "$reason" "$out/$stage-$mode.stdout"
  phase "$stage-reject-$mode" 1 "$oracle" --check "$stage" "$out/$stage-$mode.stdout"
 done
done
phase default-prepared 0 env -u SOUNIO_NV2_IR_TRACE_FULL_STAGE SOUNIO_NV2_IR_TRACE_FULL=1 timeout 30 "$component" valid
cmp "$out/default-prepared.stdout" "$out/prepared-valid.stdout" > "$out/default-prepared.verify" || bad=1
phase invalid-stage 0 env SOUNIO_NV2_IR_TRACE_FULL=1 SOUNIO_NV2_IR_TRACE_FULL_STAGE=invalid timeout 30 "$component" valid
phase invalid-stage-refusal 0 "$oracle" --refusal unknown 10 "$out/invalid-stage.stdout"
phase reject-schema1 1 "$oracle" --check prepared "$schema1"
phase reject-missing-stage 64 "$oracle" --check "$out/prepared-valid.stdout"
cat "$out/source-valid.stdout" "$out/prepared-valid.stdout" > "$out/mixed.stdout"
phase reject-mixed-stages 1 "$oracle" --check source "$out/mixed.stdout"
sha256sum -c "$out/ARTIFACTS.sha256" > "$out/ARTIFACTS.final.verify"
echo "$bad" > "$out/GATE_STATUS";test "$bad" = 0
echo NV2_FULL_TRACE_SCHEMA2_STAGE_COMPONENT_ONLY_OK > "$out/STATUS"
cat "$out/STATUS"
