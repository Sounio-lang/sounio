#!/usr/bin/env bash
# Focused scientific contract gate. Run only in a Slurm allocation.
set -euo pipefail
if test "$#" != 4;then echo 'usage: stiff_ode_contract_gate.sh RAW SOURCE ORACLE FRESH_OUTPUT' >&2;exit 64;fi
test -n "${SLURM_JOB_ID:-}" || { echo 'Slurm required' >&2;exit 64; }
raw=$(realpath "$1");source_root=$(realpath "$2");oracle=$(realpath "$3");out=$(realpath -m "$4")
test -x "$raw";test -x "$oracle";test ! -e "$out";mkdir "$out"
export SOUNIO_STDLIB_PATH="$source_root/stdlib"
ulimit -s524288
ulimit -c0
cd "$source_root"
sha256sum "$raw" "$oracle" > "$out/ARTIFACTS.sha256"
find stdlib tests/fixtures/wave4_stiff_ode -type f -print0|sort -z|xargs -0 sha256sum > "$out/SOURCE.sha256"
sha256sum tests/stdlib/epistemic/test_stiff_ode_knightian_mc.sio tests/reference/stiff_ode_contract_reference.cpp scripts/stdlib/stiff_ode_contract_gate.sh >> "$out/SOURCE.sha256"
printf 'phase\texpected\tactual\n' > "$out/RESULTS.tsv"
bad=0;last_status=0
phase(){ local label="$1" expected="$2";shift 2;last_status=0;/usr/bin/time -v "$@" > "$out/$label.stdout" 2> "$out/$label.stderr" || last_status=$?;printf '%s\t%s\t%s\n' "$label" "$expected" "$last_status" >> "$out/RESULTS.tsv";if test "$last_status" != "$expected";then bad=1;fi; }
phase independent-report 0 "$oracle" --report
# Preserve the original parser/API refusal and the previously typed captured
# callback refusal separately from the scientific counterexample.
for name in original typed_capture_refusal;do
 cp "tests/fixtures/wave4_stiff_ode/$name.sio.in" "$out/$name.sio"
 expected=1;if test "$name" = typed_capture_refusal;then expected=0;fi
 phase "$name-check" "$expected" timeout 180 "$raw" check "$out/$name.sio"
 phase "$name-compile" 1 timeout 300 "$raw" compile "$out/$name.sio" -o "$out/$name.elf"
 test ! -e "$out/$name.elf" || bad=1
done
grep -F '#1542' "$out/typed_capture_refusal-compile.stdout" > "$out/CAPTURE_DIAGNOSTIC" || bad=1
phase candidate-check 0 timeout 180 "$raw" check tests/stdlib/epistemic/test_stiff_ode_knightian_mc.sio
phase candidate-compile 0 timeout 300 "$raw" compile tests/stdlib/epistemic/test_stiff_ode_knightian_mc.sio -o "$out/candidate.elf"
if test "$last_status" = 0;then
 chmod +x "$out/candidate.elf"
 sha256sum "$out/candidate.elf" > "$out/CANDIDATE_ELF.sha256"
 phase candidate-native 0 timeout 30 "$out/candidate.elf"
 phase candidate-oracle 0 "$oracle" --check "$out/candidate-native.stdout"
fi
phase counterexample-check 0 timeout 180 "$raw" check tests/fixtures/wave4_stiff_ode/original_predicates_counterexample.sio
phase counterexample-compile 0 timeout 300 "$raw" compile tests/fixtures/wave4_stiff_ode/original_predicates_counterexample.sio -o "$out/counterexample.elf"
if test "$last_status" = 0;then
 chmod +x "$out/counterexample.elf"
 sha256sum "$out/counterexample.elf" > "$out/COUNTEREXAMPLE_ELF.sha256"
 phase counterexample-native 1 timeout 30 "$out/counterexample.elf"
 printf 'STIFF_ORIGINAL_PREDICATES 0 0 0\nSTIFF_ORIGINAL_EXPECTATIONS_COUNTEREXAMPLE\n' > "$out/counterexample.expected"
 cmp "$out/counterexample.expected" "$out/counterexample-native.stdout" > "$out/counterexample.verify" || bad=1
fi
for kind in value zero-mc missing duplicate nan;do
 phase "mutate-$kind" 0 "$oracle" --mutate "$kind" "$out/candidate-native.stdout" "$out/control-$kind.stdout"
 phase "reject-$kind" 1 "$oracle" --check "$out/control-$kind.stdout"
done
phase oracle-missing-argc 64 "$oracle"
sha256sum -c "$out/ARTIFACTS.sha256" > "$out/ARTIFACTS.final.verify"
sha256sum -c "$out/SOURCE.sha256" > "$out/SOURCE.final.verify"
if test -f "$out/CANDIDATE_ELF.sha256";then sha256sum -c "$out/CANDIDATE_ELF.sha256" > "$out/CANDIDATE_ELF.final.verify";fi
if test -f "$out/COUNTEREXAMPLE_ELF.sha256";then sha256sum -c "$out/COUNTEREXAMPLE_ELF.sha256" > "$out/COUNTEREXAMPLE_ELF.final.verify";fi
echo "$bad" > "$out/GATE_STATUS";test "$bad" = 0
echo STIFF_ODE_CPP23_REAL_MC_PBOX_GUM_CARA_GATE_OK > "$out/STATUS"
cat "$out/STATUS"
