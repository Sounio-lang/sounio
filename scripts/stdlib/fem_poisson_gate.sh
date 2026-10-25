#!/usr/bin/env bash
# Native scientific gate; execute on an allocated Linux CPU worker.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?fresh absolute output directory required}
before=${2:-}
[[ "$out" == /* && ! -e "$out" && -x ${MADAROS_RAW_BIN:?set identified raw compiler} ]]
mkdir -p "$out"
trap 'rc=$?; printf "%s\n" "$rc" > "$out/STATUS"' EXIT
cd "$root"
# The wrapper gives this inherited raw override priority over the identified compiler.
unset SOUNIO_SOUC_BIN
export SOUC_BIN="$MADAROS_RAW_BIN"
ulimit -s 524288
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_SOUC_ENGINE=madaros SOUNIO_MADAROS_NOCACHE=1
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILER.sha256"
find stdlib tests/stdlib/fem_poisson -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum tests/run-pass/{fem_poisson_checked,test_fem,test_multigrid}.sio tests/reference/fem_poisson_reference.cpp scripts/stdlib/fem_poisson_gate.sh >> "$out/INPUTS.sha256"
printf 'phase\texit_status\n' > "$out/RESULTS.tsv"
printf 'control\texpected\tactual\n' > "$out/CONTROLS.tsv"
step(){ local name=$1;shift;local rc=0;"$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?;printf '%s\t%s\n' "$name" "$rc" >> "$out/RESULTS.tsv";return "$rc"; }
expected(){ local name=$1 want=$2;shift 2;local rc=0;"$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?;printf '%s\t%s\t%s\n' "$name" "$want" "$rc" >> "$out/CONTROLS.tsv";[[ $rc == "$want" ]]; }
no_diagnostics(){ ! grep -Ei '(^|[[:space:]])error(\[|:)|typecheck: failed|parse failed' "$@" > "$out/last-diagnostic-scan.txt"; }
build(){
 local name=$1 src=$2
 if [[ -n ${SEED_RAW_BIN:-} ]];then
  step "$name.seed-check" env SOUNIO_SOUC_BIN="$SEED_RAW_BIN" SOUC_NATIVE_BIN="$SEED_RAW_BIN" scripts/ci/souc-native-wrapper.sh check "$src"
  no_diagnostics "$out/$name.seed-check.stdout" "$out/$name.seed-check.stderr"
 fi
 step "$name.check" timeout 180 bin/souc check "$src"
 no_diagnostics "$out/$name.check.stdout" "$out/$name.check.stderr"
 step "$name.compile" timeout 180 bin/souc compile "$src" -o "$out/$name.elf"
 no_diagnostics "$out/$name.compile.stdout" "$out/$name.compile.stderr"
 chmod +x "$out/$name.elf";sha256sum "$out/$name.elf" >> "$out/ELFS.sha256"
}
native(){ step "$1.native" /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$1.resource" timeout 180 "$out/$1.elf"; }
[[ -z ${SEED_RAW_BIN:-} ]] || sha256sum "$SEED_RAW_BIN" > "$out/SEED.sha256"
step reference-toolchain g++ --version
step reference-build g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/fem_poisson_reference.cpp -lquadmath -o "$out/reference"
sha256sum "$out/reference" > "$out/REFERENCE.sha256"
sha256sum "$out/reference" >> "$out/ELFS.sha256"
build fem tests/run-pass/fem_poisson_checked.sio;native fem
step fem.concordance "$out/reference" "$out/fem.native.stdout"
build poisson tests/stdlib/fem_poisson/poisson_main.sio;native poisson
step poisson.concordance "$out/reference" --poisson "$out/poisson.native.stdout"
build refinement tests/stdlib/fem_poisson/refinement_main.sio;native refinement
step refinement.concordance "$out/reference" --refinement "$out/refinement.native.stdout"
build scaled-load tests/stdlib/fem_poisson/scaled_load_main.sio;native scaled-load
step scaled-load.concordance "$out/reference" --scaled-load "$out/scaled-load.native.stdout"
for name in fem multigrid;do
 build "legacy-$name" "tests/run-pass/test_$name.sio";native "legacy-$name"
 step "legacy-$name.marker" grep -Fx 'ALL 12 TESTS PASSED' "$out/legacy-$name.native.stdout"
done
cat stdlib/math/fem.sio tests/stdlib/fem_poisson/legacy_cg_main.sio > "$out/legacy-cg.sio"
build legacy-cg "$out/legacy-cg.sio";native legacy-cg
step legacy-cg.concordance "$out/reference" --legacy-cg "$out/legacy-cg.native.stdout"
cat stdlib/math/fem.sio tests/stdlib/fem_poisson/singular_legacy_main.sio > "$out/singular.sio"
build singular "$out/singular.sio"
expected singular-refusal.native 1 timeout 30 "$out/singular.elf"
! grep -q FEM_SINGULAR_CONTINUED "$out/singular-refusal.native.stdout"
# Mutated input and verification log paths are different. Exact exit codes
# distinguish numerical/counter rejection from missing or truncated input.
awk 'NR==1 {$5=0} {print}' "$out/fem.native.stdout" > "$out/mutant-fem.input.stdout"
expected mutant-fem.verify 8 "$out/reference" "$out/mutant-fem.input.stdout"
grep -Fx 'fem-node 0 0' "$out/mutant-fem.verify.stderr"
sed '/^FEM_CHECKED_PASS$/d' "$out/fem.native.stdout" > "$out/mutant-completion.input.stdout"
expected mutant-completion.verify 9 "$out/reference" "$out/mutant-completion.input.stdout"
awk '$1=="MG"&&$2==2&&$3==4 {$4=0} {print}' "$out/poisson.native.stdout" > "$out/mutant-poisson.input.stdout"
expected mutant-poisson.verify 8 "$out/reference" --poisson "$out/mutant-poisson.input.stdout"
awk 'BEGIN{done=0} $1=="MG_WORK"&&!done {$5=$5+1;done=1} {print}' "$out/poisson.native.stdout" > "$out/mutant-work.input.stdout"
expected mutant-work.verify 13 "$out/reference" --poisson "$out/mutant-work.input.stdout"
if [[ -n "$before" ]];then
 [[ "$before" == /* && -f "$before/math/fem.sio" && -f "$before/math/fem_checked.sio" ]]
 sha256sum "$before/math/fem.sio" "$before/math/fem_checked.sio" > "$out/BEFORE.sha256"
 cat "$before/math/fem.sio" tests/stdlib/fem_poisson/legacy_cg_main.sio > "$out/legacy-cg-before.sio"
 build legacy-cg-before "$out/legacy-cg-before.sio"
 expected legacy-cg-before.native 1 timeout 30 "$out/legacy-cg-before.elf"
 expected legacy-cg-before.verify 8 "$out/reference" --legacy-cg "$out/legacy-cg-before.native.stdout"
 cat "$before/math/fem.sio" tests/stdlib/fem_poisson/singular_legacy_main.sio > "$out/singular-before.sio"
 build singular-before "$out/singular-before.sio"
 step singular-before.native timeout 30 "$out/singular-before.elf"
 step singular-before.marker grep -Fx FEM_SINGULAR_CONTINUED "$out/singular-before.native.stdout"
 expected singular-before.verify 8 "$out/reference" --singular-before "$out/singular-before.native.stdout"
 SOUNIO_STDLIB_PATH="$before" build scaled-load-before tests/stdlib/fem_poisson/scaled_load_main.sio
 expected scaled-load-before.native 1 timeout 30 "$out/scaled-load-before.elf"
 expected scaled-load-before.verify 8 "$out/reference" --scaled-load "$out/scaled-load-before.native.stdout"
fi
step input-integrity sha256sum -c "$out/INPUTS.sha256"
step compiler-integrity sha256sum -c "$out/COMPILER.sha256"
step executable-integrity sha256sum -c "$out/ELFS.sha256"
echo FEM_POISSON_GATE_PASS
