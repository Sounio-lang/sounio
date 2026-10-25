#!/usr/bin/env bash
# Bounded64 production-import numerical/contract gate. Run on a UID1000 Linux worker.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $(uname -s) == Linux && $(id -u) == 1000 ]] || { echo 'Linux UID1000 worker required' >&2; exit 2; }
: "${MADAROS_RAW_BIN:?absolute verified modular compiler required}"
: "${DP54_GATE_OUTPUT:?new absolute output directory required}"
[[ $MADAROS_RAW_BIN == /* && -x $MADAROS_RAW_BIN && $DP54_GATE_OUTPUT == /* ]] || exit 2
[[ ! -e $DP54_GATE_OUTPUT ]] || { echo 'output must not exist' >&2; exit 2; }
mkdir -p "$DP54_GATE_OUTPUT"
out=$DP54_GATE_OUTPUT
ulimit -s 524288
ulimit -c 0
export LC_ALL=C LANG=C SOUNIO_STDLIB_PATH="$PWD/stdlib"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
engines=(madaros)
if [[ -n ${DP54_SEED_BIN:-} ]]; then
 [[ $DP54_SEED_BIN == /* && -x $DP54_SEED_BIN ]] || exit 2
 export SOUC_NATIVE_BIN=$DP54_SEED_BIN
 engines+=(seed)
fi
printf '%s\n' "${engines[@]}" > "$out/ENGINES.txt"
printf 'engine\tcase\tphase\tstatus\n' > "$out/RESULTS.tsv"
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILERS.sha256"
if [[ -n ${DP54_SEED_BIN:-} ]]; then sha256sum "$DP54_SEED_BIN" >> "$out/COMPILERS.sha256"; fi
find bin/souc bin/madaros scripts/lib scripts/ci/souc-native-wrapper.sh \
 scripts/stdlib/dp54_dense_events_gate.sh stdlib/ode stdlib/math \
 tests/run-pass tests/compile-fail tests/stdlib/ode tests/reference \
 docs/evidence/stdlib-SIO-174/before/callback_time.sio \
 -type f -print0 | sort -z | xargs -0 sha256sum > "$out/SOURCE.sha256"
run_phase() {
 local engine=$1 name=$2 phase=$3; shift 3
 local rc=0
 "$@" > "$out/$engine.$name.$phase.stdout" 2> "$out/$engine.$name.$phase.stderr" || rc=$?
 printf '%s\n' "$rc" > "$out/$engine.$name.$phase.status"
 printf '%s\t%s\t%s\t%s\n' "$engine" "$name" "$phase" "$rc" >> "$out/RESULTS.tsv"
 return "$rc"
}
run_phase cpp reference compile g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/dp54_dense_events_reference.cpp -o "$out/reference"
run_phase cpp reference selftest "$out/reference" selftest
run_phase cpp legacy compile g++ -std=c++23 -O2 -Wall -Wextra -pedantic tests/reference/deep_ode_reference.cpp -o "$out/emitter"
"$out/emitter" > "$out/legacy_reference.sio"
sha256sum "$out/legacy_reference.sio" > "$out/GENERATED.sha256"
positive() {
 local engine=$1 name=$2 src=$3 marker=$4
 if [[ $engine == madaros ]]; then
  run_phase "$engine" "$name" check timeout 180 bin/souc check "$src"
  run_phase "$engine" "$name" compile timeout 240 bin/souc compile "$src" -o "$out/$engine.$name.elf"
 else
  run_phase "$engine" "$name" check timeout 180 bash scripts/ci/souc-native-wrapper.sh check "$src"
  run_phase "$engine" "$name" compile timeout 240 "$DP54_SEED_BIN" "$src" "$out/$engine.$name.elf"
 fi
 ! grep -Eiq 'error\[E[0-9]+\]|typecheck:.*failed|parse error' "$out/$engine.$name."{check,compile}.{stdout,stderr}
 [[ -s $out/$engine.$name.elf ]]
 chmod +x "$out/$engine.$name.elf"
 run_phase "$engine" "$name" native /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$engine.$name.resource" timeout 120 "$out/$engine.$name.elf"
 grep -Fxq "$marker" "$out/$engine.$name.native.stdout"
 awk -F= '$1=="elapsed_seconds"&&$2>15{exit 1}$1=="max_rss_kib"&&$2>131072{exit 1}' "$out/$engine.$name.resource"
 sha256sum "$out/$engine.$name.elf" > "$out/$engine.$name.ELF.sha256"
}
for engine in "${engines[@]}"; do
 positive "$engine" core tests/run-pass/stdlib_dp54_dense_events.sio STDLIB_DP54_DENSE_EVENTS_PASS
 positive "$engine" generic tests/run-pass/stdlib_deep_ode_generic.sio STDLIB_DEEP_ODE_GENERIC_PASS
 positive "$engine" status tests/run-pass/stdlib_deep_ode_status.sio STDLIB_DEEP_ODE_STATUS_PASS
 positive "$engine" analytic "$out/legacy_reference.sio" STDLIB_DEEP_ODE_REFERENCE_PASS
 positive "$engine" invalid tests/stdlib/ode/dp54_invalid_probe.sio DP54_INVALID_PASS
 positive "$engine" baseline docs/evidence/stdlib-SIO-174/before/callback_time.sio DP54_BASELINE_DONE
 run_phase "$engine" core reference "$out/reference" "$out/$engine.core.native.stdout"
 run_phase "$engine" invalid reference "$out/reference" invalid "$out/$engine.invalid.native.stdout"
 run_phase "$engine" baseline reference "$out/reference" after "$out/$engine.baseline.native.stdout"
 for control in value missing linear_value linear_slope counter bracket nonfinite duplicate marker; do
  src="$out/$engine.core.native.stdout"; dst="$out/$engine.control.$control.input"
  case $control in
   value) awk 'BEGIN{done=0}$1=="DP54_VALUE"&&!done{$3=0;done=1}{print}' "$src" > "$dst";;
   missing) awk '!($1=="DP54_VALUE"&&$2=="constant.end")' "$src" > "$dst";;
   linear_value) awk '$1=="DP54_POINT"&&$2==3{$4="4602678819172646912"}{print}' "$src" > "$dst";;
   linear_slope) awk '$1=="DP54_POINT"&&$2==3{$5="4607182418800017408"}{print}' "$src" > "$dst";;
   counter) awk '$1=="DP54_META"&&$2=="constant"{$6=28}{print}' "$src" > "$dst";;
   bracket) awk '$1=="DP54_VALUE"&&$2=="event.terminal.lo"{$3="4607182418800017408"}{print}' "$src" > "$dst";;
   nonfinite) awk 'BEGIN{done=0}$1=="DP54_VALUE"&&!done{$3="9218868437227405312";done=1}{print}' "$src" > "$dst";;
   duplicate) cat "$src" > "$dst"; awk '$1=="DP54_VALUE"{print;exit}' "$src" >> "$dst";;
   marker) sed '/^STDLIB_DP54_DENSE_EVENTS_PASS$/d' "$src" > "$dst";;
  esac
  rc=0; run_phase "$engine" "control.$control" reference "$out/reference" "$dst" || rc=$?
  [[ $rc == 1 ]]
 done
 for form in dense_private event_private dense_linear_reuse; do
  src="tests/compile-fail/dp54_$form.sio"; rc=0
  grep -q "^//@ error-pattern: " "$src"
  ! grep -q "^//@ expect-error:" "$src"
  if [[ $engine == madaros ]]; then
   run_phase "$engine" "$form" refusal timeout 180 bin/souc check "$src" || rc=$?
  else
   run_phase "$engine" "$form" refusal timeout 180 bash scripts/ci/souc-native-wrapper.sh check "$src" || rc=$?
  fi
  [[ $rc != 0 && $rc != 124 ]]
  if [[ $form == dense_linear_reuse ]]; then
   grep -Eiq 'E039|linear value.*more than once' "$out/$engine.$form.refusal."{stdout,stderr}
  else
   grep -Eiq 'E259|private field' "$out/$engine.$form.refusal."{stdout,stderr}
  fi
 done
done
sha256sum -c "$out/SOURCE.sha256" > "$out/SOURCE.verify.txt"
sha256sum -c "$out/COMPILERS.sha256" > "$out/COMPILERS.verify.txt"
sha256sum -c "$out/GENERATED.sha256" > "$out/GENERATED.verify.txt"
echo DP54_DENSE_EVENTS_GATE_PASS | tee "$out/FINAL.txt"
