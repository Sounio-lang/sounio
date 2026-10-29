#!/usr/bin/env bash
# Formatter-only component gate. Requires an allocation; does no compiler build.
set -euo pipefail
test "$#" = 3 || { echo 'usage: nv2_trace_full_gate.sh COMPONENT ORACLE FRESH_OUTPUT' >&2;exit 64; }
test -n "${SLURM_JOB_ID:-}" || { echo 'Slurm required' >&2;exit 64; }
component=$(realpath "$1");oracle=$(realpath "$2");out=$(realpath -m "$3")
test -x "$component";test -x "$oracle";test ! -e "$out";mkdir -p "$out"
sha256sum "$component" "$oracle" > "$out/ARTIFACTS.sha256"
printf 'phase\texpected\tactual\n' > "$out/RESULTS.tsv"
bad=0;last=0
phase(){ local name="$1" expected="$2";shift 2;last=0;/usr/bin/time -v "$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || last=$?;printf '%s\t%s\t%s\n' "$name" "$expected" "$last" >> "$out/RESULTS.tsv";if test "$last" != "$expected";then bad=1;fi; }
phase valid 0 env SOUNIO_NV2_IR_TRACE_FULL=1 timeout 30 "$component" valid
phase oracle-valid 0 "$oracle" --probe "$out/valid.stdout"
phase legacy 0 env -u SOUNIO_NV2_IR_TRACE_FULL timeout 30 "$component" valid
printf 'LEGACY_CONTROL_BYTES\n' > "$out/legacy.expected"
cmp "$out/legacy.expected" "$out/legacy.stdout" > "$out/legacy.verify" || bad=1
for control in truncate duplicate total args bits name float-word unknown end;do
 phase "mutate-$control" 0 "$oracle" --mutate "$control" "$out/valid.stdout" "$out/control-$control.stdout"
 phase "reject-$control" 1 "$oracle" --probe "$out/control-$control.stdout"
done
for pair in badargs:8 badname:7 badregion:6 badparam:4 budget:5;do
 mode="${pair%:*}";reason="${pair#*:}"
 phase "$mode" 0 env SOUNIO_NV2_IR_TRACE_FULL=1 timeout 30 "$component" "$mode"
 phase "refusal-$mode" 0 "$oracle" --refusal "$reason" "$out/$mode.stdout"
 phase "reject-$mode" 1 "$oracle" --check "$out/$mode.stdout"
done
sha256sum -c "$out/ARTIFACTS.sha256" > "$out/ARTIFACTS.final.verify"
echo "$bad" > "$out/GATE_STATUS";test "$bad" = 0
echo NV2_FULL_TRACE_FORMATTER_COMPONENT_ONLY_OK > "$out/STATUS"
cat "$out/STATUS"
