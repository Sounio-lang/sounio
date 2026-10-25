#!/usr/bin/env bash
# Native Linux only, explicit source-built compiler, fresh artifact directory.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${TRUST_GATE_OUT:?fresh absolute output path required}
[[ "$out" == /* && ! -e "$out" && -x "${MADAROS_RAW_BIN:?explicit executable compiler required}" ]]
mkdir -p "$out"
trap 'rc=$?; printf "%s\n" "$rc" > "$out/STATUS"' EXIT
export PATH=/usr/bin:/bin LANG=C LC_ALL=C SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_STDLIB_PATH="$repo/stdlib" SOUNIO_MADAROS_NOCACHE=1
ulimit -s 524288
cd "$repo"
printf 'phase\texit_status\n' > "$out/RESULTS.tsv"
step(){ local name=$1; shift; local rc=0; "$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?; printf '%s\t%s\n' "$name" "$rc" >> "$out/RESULTS.tsv"; return "$rc"; }
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILER.sha256"
find stdlib -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum tests/run-pass/ontology_trust_semiring.sio tests/run-pass/ontology_trust_boundary.sio tests/compile-fail/ontology_trust_private.sio tests/reference/ontology_trust_reference.cpp scripts/stdlib/ontology_trust_gate.sh >> "$out/INPUTS.sha256"
step cpp-build timeout 120 g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/ontology_trust_reference.cpp -o "$out/reference"
step generate "$out/reference" --generate "$out/generated.sio"
step generator-identity cmp tests/run-pass/ontology_trust_semiring.sio "$out/generated.sio"
for name in semiring boundary; do
 src="tests/run-pass/ontology_trust_$name.sio"
 step "$name-check" timeout 120 bin/souc check "$src"
 step "$name-compile" timeout 180 bin/souc compile "$src" -o "$out/$name.elf"
 chmod +x "$out/$name.elf"
 step "$name-native" /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.resource" timeout --kill-after=5 300 "$out/$name.elf"
done
step boundary-marker grep -Fx TRUST_BOUNDARY_PASS "$out/boundary-native.stdout"
step concordance "$out/reference" --compare "$out/semiring-native.stdout"
rc=0
timeout 120 bin/souc check tests/compile-fail/ontology_trust_private.sio > "$out/private.stdout" 2> "$out/private.stderr" || rc=$?
printf 'private-constructor-rejected\t%s\n' "$rc" >> "$out/CONTROLS.tsv"
[[ "$rc" == 1 ]]
grep -qi private "$out/private.stdout" "$out/private.stderr"
# Controls retain syntactically valid rows but falsify identity/completeness.
sed 's/^TRUST_FACT 0 0 0 0 0$/TRUST_FACT 0 0 1 1 1/' "$out/semiring-native.stdout" > "$out/wrong-fact.stdout"
grep -v '^TRUST_SEMIRING_ELPLUS_PASS$' "$out/semiring-native.stdout" > "$out/missing-marker.stdout"
for name in wrong-fact missing-marker; do
 rc=0
 "$out/reference" --compare "$out/$name.stdout" > "$out/$name.check.stdout" 2> "$out/$name.check.stderr" || rc=$?
 printf '%s-rejected\t%s\n' "$name" "$rc" >> "$out/CONTROLS.tsv"
 [[ "$rc" == 1 ]]
done
# Source mutants test the actual algebra and weighted rule binding.
mkdir "$out/mutant"
cp -a stdlib "$out/mutant/stdlib"
sed 's/if a < b { a } else { b }/a/' stdlib/ontology/trust_semiring.sio > "$out/mutant/stdlib/ontology/trust_semiring.sio"
step scalar-mutant-compile env SOUNIO_STDLIB_PATH="$out/mutant/stdlib" timeout 180 bin/souc compile tests/run-pass/ontology_trust_semiring.sio -o "$out/scalar-mutant.elf"
chmod +x "$out/scalar-mutant.elf"
rc=0
timeout 30 "$out/scalar-mutant.elf" > "$out/scalar-mutant.stdout" 2> "$out/scalar-mutant.stderr" || rc=$?
printf 'scalar-mutant-rejected\t%s\n' "$rc" >> "$out/CONTROLS.tsv"
[[ "$rc" == 13 ]]
cp stdlib/ontology/trust_semiring.sio "$out/mutant/stdlib/ontology/trust_semiring.sio"
sed 's/,36928+k) != 0/,0-1) != 0/' stdlib/ontology/elplus_trust.sio > "$out/mutant/stdlib/ontology/elplus_trust.sio"
cmp -s stdlib/ontology/elplus_trust.sio "$out/mutant/stdlib/ontology/elplus_trust.sio" && exit 65
step chain-mutant-compile env SOUNIO_STDLIB_PATH="$out/mutant/stdlib" timeout 180 bin/souc compile tests/run-pass/ontology_trust_semiring.sio -o "$out/chain-mutant.elf"
chmod +x "$out/chain-mutant.elf"
step chain-mutant-native timeout 300 "$out/chain-mutant.elf"
rc=0
"$out/reference" --compare "$out/chain-mutant-native.stdout" > "$out/chain-mutant.check.stdout" 2> "$out/chain-mutant.check.stderr" || rc=$?
printf 'chain-premise-mutant-rejected\t%s\n' "$rc" >> "$out/CONTROLS.tsv"
[[ "$rc" == 1 ]]
sha256sum "$out"/*.elf "$out/reference" > "$out/EXECUTABLES.sha256"
step input-integrity sha256sum -c "$out/INPUTS.sha256"
step compiler-integrity sha256sum -c "$out/COMPILER.sha256"
echo ONTOLOGY_TRUST_GATE_PASS
