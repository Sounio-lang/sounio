#!/usr/bin/env bash
# Both actual native engines, independent C++ framing oracle, reusable output.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?fresh absolute output required}
[[ "$out" == /* && -x ${MADAROS_RAW_BIN:?} && -x ${SEED_RAW_BIN:?} ]] || exit 64
mkdir -p "$(dirname "$out")"
mkdir "$out" || exit 65
trap 'rc=$?;printf "%s\n" "$rc" > "$out/STATUS"' EXIT
cd "$root"
unset SOUNIO_SOUC_BIN
export SOUC_BIN="$MADAROS_RAW_BIN" SOUNIO_SOUC_ENGINE=madaros SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_MADAROS_NOCACHE=1
ulimit -s 524288
sha256sum "$MADAROS_RAW_BIN" "$SEED_RAW_BIN" > "$out/COMPILERS.sha256"
find stdlib -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum tests/reference/{genomics_named_stream,genomics_count_identity}_reference.cpp tests/stdlib/genomics_counts/{named_slot,named_smoke}.sio scripts/stdlib/genomics_named_stream_gate.sh >> "$out/INPUTS.sha256"
printf 'phase\texit\n' > "$out/RESULTS.tsv"
step(){ local name=$1;shift;local rc=0;"$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?;printf '%s\t%s\n' "$name" "$rc" >> "$out/RESULTS.tsv";return "$rc"; }
build_both(){
 local name=$1 src=$2
 sha256sum "$src" >> "$out/CALLERS.sha256"
 step "$name.check" timeout 180 bin/souc check "$src"
 step "$name.compile" timeout 180 bin/souc compile "$src" -o "$out/$name.elf"
 chmod +x "$out/$name.elf"
 step "$name.native" /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.resource" timeout 300 "$out/$name.elf"
 step "$name.seed-compile" timeout 180 "$SEED_RAW_BIN" "$src" "$out/$name.seed.elf"
 chmod +x "$out/$name.seed.elf"
 step "$name.seed-native" /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.seed.resource" timeout 300 "$out/$name.seed.elf"
 sha256sum "$out/$name.elf" "$out/$name.seed.elf" >> "$out/ELFS.sha256"
}
for stem in genomics_named_stream genomics_count_identity;do
 step "$stem.build" g++ -std=c++23 -O2 -Wall -Wextra -Werror "tests/reference/${stem}_reference.cpp" -o "$out/$stem.reference"
 sha256sum "$out/$stem.reference" >> "$out/REFERENCES.sha256"
done
build_both slot tests/stdlib/genomics_counts/named_slot.sio
build_both smoke tests/stdlib/genomics_counts/named_smoke.sio
step generate "$out/genomics_count_identity.reference" --generate-named
cp "$out/generate.stdout" "$out/legacy.sio"
step adapt "$out/genomics_named_stream.reference" --adapt-named "$out/legacy.sio"
cp "$out/adapt.stdout" "$out/into.sio"
for mode in legacy into;do
 build_both "$mode" "$out/$mode.sio"
 for engine in native seed-native;do
  step "$mode.$engine.verify" "$out/genomics_count_identity.reference" --named "$out/$mode.$engine.stdout"
 done
done
for engine in native seed-native;do
 step "slot.$engine.marker" grep -Fxq NAMED_SLOT_PASS "$out/slot.$engine.stdout"
 step "smoke.$engine.marker" grep -Fxq NAMED_COUNT_SMOKE_PASS "$out/smoke.$engine.stdout"
 step "same.$engine" cmp "$out/legacy.$engine.stdout" "$out/into.$engine.stdout"
done
for size in 100000 1000000;do
 step "bench$size.generate" "$out/genomics_named_stream.reference" --generate "$size" into
 cp "$out/bench$size.generate.stdout" "$out/bench$size.sio"
 build_both "bench$size" "$out/bench$size.sio"
 for engine in native seed-native;do
  step "bench$size.$engine.verify" "$out/genomics_named_stream.reference" --verify "$size" "$out/bench$size.$engine.stdout"
 done
done
# Independent corrupted inputs, distinct paths from verifier stdout/stderr.
awk '$1=="NAMED_BENCH"{$4=0} {print}' "$out/bench100000.native.stdout" > "$out/mutant-position.input"
rc=0;step mutant-position "$out/genomics_named_stream.reference" --verify 100000 "$out/mutant-position.input" || rc=$?
[[ $rc == 8 ]];grep -Fq 'named stream count/identity/position disagreement' "$out/mutant-position.stderr"
sed '/^NAMED_BENCH_PASS$/d' "$out/bench100000.native.stdout" > "$out/mutant-completion.input"
rc=0;step mutant-completion "$out/genomics_named_stream.reference" --verify 100000 "$out/mutant-completion.input" || rc=$?
[[ $rc == 9 ]]
step unchanged sha256sum -c "$out/INPUTS.sha256"
echo NAMED_STREAM_GATE_PASS
