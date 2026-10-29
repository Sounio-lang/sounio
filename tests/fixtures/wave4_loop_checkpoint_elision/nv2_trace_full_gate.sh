#!/usr/bin/env bash
# Stage-bound schema-2 formatter component; no compiler bootstrap.
set -euo pipefail
test "$#" = 4 || { echo 'usage: nv2_trace_full_gate.sh COMPONENT ORACLE SCHEMA1_LOG FRESH_OUTPUT' >&2;exit 64; }
test -n "${SLURM_JOB_ID:-}" || { echo 'Slurm required' >&2;exit 64; }
component=$(realpath "$1");oracle=$(realpath "$2");schema1=$(realpath "$3");out=$(realpath -m "$4")
test -x "$component";test -x "$oracle";test -s "$schema1";test ! -e "$out";mkdir -p "$out"
sha256sum "$component" "$oracle" "$schema1" > "$out/ARTIFACTS.sha256"
printf 'phase\texpected\tactual\n' > "$out/RESULTS.tsv"
bad=0;last=0
phase(){ local name="$1" expected="$2";shift 2;last=0;/usr/bin/time -v "$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || last=$?;printf '%s\t%s\t%s\n' "$name" "$expected" "$last" >> "$out/RESULTS.tsv";if test "$last" != "$expected";then bad=1;fi; }
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
