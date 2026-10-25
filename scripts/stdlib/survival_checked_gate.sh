#!/usr/bin/env bash
# Allocated Linux CPU qualification, actual dual-engine execution and C++23.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: survival_checked_gate.sh FRESH_OUTPUT_DIRECTORY}"
: "${MADAROS_RAW_BIN:?explicit source-built compiler required}"
[[ -z "${SOUNIO_SOUC_BIN:-}" ]] || { echo 'raw override bypasses engine routing' >&2;exit 64; }
mkdir -p "$(dirname "$OUT")"
if ! mkdir "$OUT";then echo "fresh output directory required: $OUT" >&2;exit 65;fi
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
ulimit -s 524288
id > "$OUT/identity.txt"
sha256sum "$MADAROS_RAW_BIN" bin/souc-lean-single-x86_64 > "$OUT/compiler.sha256"
sha256sum stdlib/stats/{checked_survival,logrank,checked_distributions}.sio stdlib/medical/survival.sio \
 stdlib/collections/{numeric_index_sort,numeric_buffer,integer_buffer}.sio stdlib/math/{pure,logarithmic}.sio \
 stdlib/special/checked_gamma_beta.sio tests/reference/survival_checked_reference.cpp \
 tests/run-pass/stdlib_survival_checked.sio tests/stdlib/medical/test_survival_e2e.sio \
 scripts/stdlib/survival_checked_gate.sh > "$OUT/inputs.sha256"
g++ --version > "$OUT/reference.compiler.txt"
/usr/bin/time -v g++ -std=c++23 -O2 -Wall -Wextra tests/reference/survival_checked_reference.cpp -o "$OUT/reference" > "$OUT/reference.compile.stdout" 2> "$OUT/reference.compile.stderr"
sha256sum "$OUT/reference" > "$OUT/reference.sha256"
no_diagnostics() {
 if grep -Eq '^error:|error\[E' "$1" "$2";then echo "compiler diagnostics: $1" >&2;return 1;fi
}
run_fixture() {
 local src="$1" name="$2" expected="$3"
 sha256sum "$src" > "$OUT/$name.source.sha256"
 SOUNIO_SOUC_ENGINE=lean_single bin/souc check "$src" > "$OUT/$name.seed.check.stdout" 2> "$OUT/$name.seed.check.stderr"
 no_diagnostics "$OUT/$name.seed.check.stdout" "$OUT/$name.seed.check.stderr"
 for engine in lean_single madaros;do
  /usr/bin/time -v env SOUNIO_SOUC_ENGINE="$engine" bin/souc compile "$src" -o "$OUT/$name.$engine.elf" > "$OUT/$name.$engine.compile.stdout" 2> "$OUT/$name.$engine.compile.stderr"
  no_diagnostics "$OUT/$name.$engine.compile.stdout" "$OUT/$name.$engine.compile.stderr"
  chmod +x "$OUT/$name.$engine.elf";sha256sum "$OUT/$name.$engine.elf" > "$OUT/$name.$engine.elf.sha256"
  set +e
  /usr/bin/time -v timeout 900 "$OUT/$name.$engine.elf" > "$OUT/$name.$engine.native.stdout" 2> "$OUT/$name.$engine.native.stderr"
  local code=$?;echo "$code" > "$OUT/$name.$engine.native.exit"
  set -e
  [[ "$code" == "$expected" ]]
 done
 sha256sum -c "$OUT/$name.source.sha256" > "$OUT/$name.source.verify.stdout"
}
run_fixture tests/run-pass/stdlib_survival_checked.sio smoke 0
for engine in lean_single madaros;do grep -qx STDLIB_SURVIVAL_CHECKED_PASS "$OUT/smoke.$engine.native.stdout";done
"$OUT/reference" --generate > "$OUT/small.generated.sio"
run_fixture "$OUT/small.generated.sio" small 0
"$OUT/reference" --generate-scale > "$OUT/scale.generated.sio"
run_fixture "$OUT/scale.generated.sio" scale 0
"$OUT/reference" --generate-tail > "$OUT/tail.generated.sio"
run_fixture "$OUT/tail.generated.sio" tail 0
"$OUT/reference" --generate-legacy > "$OUT/legacy.generated.sio"
run_fixture "$OUT/legacy.generated.sio" legacy 0
for engine in lean_single madaros;do
 "$OUT/reference" "$OUT/small.$engine.native.stdout" > "$OUT/small.$engine.verify.stdout" 2> "$OUT/small.$engine.verify.stderr"
 "$OUT/reference" --scale "$OUT/scale.$engine.native.stdout" > "$OUT/scale.$engine.verify.stdout" 2> "$OUT/scale.$engine.verify.stderr"
 "$OUT/reference" --tail "$OUT/tail.$engine.native.stdout" > "$OUT/tail.$engine.verify.stdout" 2> "$OUT/tail.$engine.verify.stderr"
 "$OUT/reference" --legacy "$OUT/legacy.$engine.native.stdout" > "$OUT/legacy.$engine.verify.stdout" 2> "$OUT/legacy.$engine.verify.stderr"
done
run_fixture stdlib/stats/logrank.sio legacy_module 0
run_fixture tests/stdlib/medical/test_survival_e2e.sio medical_api 0
for engine in lean_single madaros;do
 grep -qx 'ALL PASS' "$OUT/legacy_module.$engine.native.stdout"
 grep -qx MEDICAL_SURVIVAL_OK "$OUT/medical_api.$engine.native.stdout"
done
# Public legacy undefined outputs refuse; all dynamic availability stays explicit.
for condition in 'n=257' 'e[0]=2' 'g[0]=1';do
 case "$condition" in
  n=257) name=shape;;
  'e[0]=2') name=event;;
  *) name=absent;;
 esac
 cat > "$OUT/$name.generated.sio" <<SIO
use stats::logrank::{logrank}
fn main()->i32 with Alloc,Mut,Div,Panic {
 var t:[f64;256]=[1.0;256];var e:[i64;256]=[1;256];var g:[i64;256]=[1;256]
 var n:i32=2
 g[0]=0
 $condition
 let r=logrank(&t,&e,&g,n)
 if r.p>0.0 {0}else{2}
}
SIO
 run_fixture "$OUT/$name.generated.sio" "$name" 1
 for engine in lean_single madaros;do grep -qi 'assert' "$OUT/$name.$engine.native.stderr";done
done
awk '$1=="CURVE"&&$2==5&&$3==0&&$4==0 {$5=0} {print}' "$OUT/small.madaros.native.stdout" > "$OUT/mutant_risk.stdout"
awk '$1=="CURVE"&&$2==5&&$3==0&&$4==0 {$14=0} {print}' "$OUT/small.madaros.native.stdout" > "$OUT/mutant_ci.stdout"
awk '$1=="CURVE"&&$2==5&&$3==0&&$4==0 {$8=0} {print}' "$OUT/small.madaros.native.stdout" > "$OUT/mutant_flags.stdout"
awk '$1=="LOGRANK"&&$2==26 {$12=0} {print}' "$OUT/small.madaros.native.stdout" > "$OUT/mutant_tail.stdout"
awk '$1=="AGGREGATE"&&$2==1 {$3=0} {print}' "$OUT/scale.madaros.native.stdout" > "$OUT/mutant_digest.stdout"
sed '/^SURVIVAL_ROWS_COMPLETE /d' "$OUT/small.madaros.native.stdout" > "$OUT/mutant_missing.stdout"
reject() {
 local name="$1" mode="$2" code="$3" diagnostic="$4"
 set +e
 if [[ -n "$mode" ]];then "$OUT/reference" "$mode" "$OUT/mutant_$name.stdout" > "$OUT/mutant_$name.verify.stdout" 2> "$OUT/mutant_$name.verify.stderr"
 else "$OUT/reference" "$OUT/mutant_$name.stdout" > "$OUT/mutant_$name.verify.stdout" 2> "$OUT/mutant_$name.verify.stderr";fi
 local actual=$?;echo "$actual" > "$OUT/mutant_$name.verify.exit"
 set -e
 [[ "$actual" == "$code" ]]
 grep -Fq "$diagnostic" "$OUT/mutant_$name.verify.stderr"
}
reject risk '' 1 'CURVE cohort=5 risk/availability row=0 group=0'
reject ci '' 1 'CURVE cohort=5 field=5 numerical'
reject flags '' 1 'CURVE cohort=5 risk/availability row=0 group=0'
reject tail '' 1 'LOGRANK cohort=26 p numerical'
reject digest --scale 1 'AGGREGATE cohort=1 full integer risk-table digest'
reject missing '' 10 SURVIVAL_REFERENCE_INCOMPLETE_OUTPUT
sha256sum -c "$OUT/inputs.sha256" > "$OUT/inputs.verify.stdout"
sha256sum -c "$OUT/compiler.sha256" > "$OUT/compiler.verify.stdout"
sha256sum -c "$OUT/reference.sha256" > "$OUT/reference.verify.stdout"
printf 'SURVIVAL_CHECKED_GATE_PASS\n' | tee "$OUT/gate.result"
