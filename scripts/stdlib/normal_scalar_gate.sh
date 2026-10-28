#!/usr/bin/env bash
# Numeric and compatibility acceptance; WorkerFn final-IR certification is separate.
set -euo pipefail
: "${SLURM_JOB_ID:?run this qualification under Slurm}"
: "${MADAROS_RAW_BIN:?explicit raw compiler required}"
: "${NORMAL_SCALAR_COMPILER_SHA256:?pin the raw compiler}"
: "${NORMAL_SCALAR_BOOST_ROOT:?directory containing the private or system boost headers}"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?absolute fresh output directory required}
case "$OUT" in /*) ;;*) exit 64;;esac
test ! -e "$OUT";mkdir -p "$OUT"
trap 'r=$?;echo "$r" > "$OUT/STATUS";exit "$r"' EXIT
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib" SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_REQUIRE_COMMITTED_MADAROS=0 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export CPLUS_INCLUDE_PATH="$NORMAL_SCALAR_BOOST_ROOT${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
ulimit -s524288
ulimit -c0
test -s "$NORMAL_SCALAR_BOOST_ROOT/boost/version.hpp"
printf '%s  %s\n' "$NORMAL_SCALAR_COMPILER_SHA256" "$MADAROS_RAW_BIN" > "$OUT/COMPILER.sha256"
sha256sum -c "$OUT/COMPILER.sha256" > "$OUT/COMPILER.before.verify"
find stdlib -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE.sha256"
sha256sum tests/reference/normal_scalar_reference.cpp tests/stdlib/wave3-core/normal_scalar.sio scripts/stdlib/normal_scalar_gate.sh >> "$OUT/SOURCE.sha256"
find "$NORMAL_SCALAR_BOOST_ROOT/boost" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/BOOST_HEADERS.sha256"
g++ --version > "$OUT/CXX_VERSION.txt"
printf 'phase\texit_status\n' > "$OUT/RESULTS.tsv"
run(){ local name=$1;shift;local r=0;"$@" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr" || r=$?;printf '%s\t%s\n' "$name" "$r" >> "$OUT/RESULTS.tsv";return "$r";}
reject(){ local name=$1;shift;local r=0;"$@" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr" || r=$?;printf '%s\t%s\n' "$name" "$r" >> "$OUT/NEGATIVES.tsv";test "$r" -ne 0;grep -F NORMAL_SCALAR_FAIL "$OUT/$name.stderr" > "$OUT/$name.reason.verify";}
for file in stdlib/special/gamma_scalar_core.sio stdlib/math/scalar_log.sio stdlib/stats/normal_scalar.sio stdlib/special/gamma.sio stdlib/special/checked_gamma_beta.sio stdlib/math/logarithmic.sio stdlib/stats/checked_distributions.sio tests/stdlib/wave3-core/normal_scalar.sio;do
 name=$(printf '%s' "$file" | tr '/.' '__')
 run "$name-check" timeout 180 bin/souc check "$file"
done
run static-compile timeout 300 bin/souc compile tests/stdlib/wave3-core/normal_scalar.sio -o "$OUT/static.elf"
run static-native timeout 120 "$OUT/static.elf"
grep -Fx NORMAL_SCALAR_OK "$OUT/static-native.stdout" > "$OUT/static.marker.verify"
run reference-compile g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/normal_scalar_reference.cpp -o "$OUT/reference"
run generate "$OUT/reference" --generate
cp "$OUT/generate.stdout" "$OUT/generated.sio"
run generated-check timeout 180 bin/souc check "$OUT/generated.sio"
run generated-compile timeout 300 bin/souc compile "$OUT/generated.sio" -o "$OUT/generated.elf"
run generated-native timeout 180 "$OUT/generated.elf"
run concordance "$OUT/reference" "$OUT/generated-native.stdout"
run application-precision "$OUT/reference" --application "$OUT/generated-native.stdout"
printf 'case\texit_status\n' > "$OUT/NEGATIVES.tsv"
run make-wrong-complement "$OUT/reference" --wrong-complement "$OUT/generated-native.stdout"
reject wrong-complement "$OUT/reference" "$OUT/make-wrong-complement.stdout"
sed '/^NORMAL_SCALAR_ROWS /d' "$OUT/generated-native.stdout" > "$OUT/missing-completion.txt"
reject missing-completion "$OUT/reference" "$OUT/missing-completion.txt"
{ cat "$OUT/generated-native.stdout";sed -n '/^ROW 0 /p' "$OUT/generated-native.stdout";} > "$OUT/duplicate.txt"
reject duplicate "$OUT/reference" "$OUT/duplicate.txt"
# Preserve the original113-bit oracle, fixtures, assertions, invalid/underflow
# refusal probes and tolerance policy through its unchanged complete gate.
run original-checked-gate bash scripts/stdlib/checked_distributions_gate.sh "$OUT/original-checked"
sha256sum "$OUT/static.elf" "$OUT/generated.elf" "$OUT/reference" > "$OUT/EXECUTABLES.sha256"
sha256sum -c "$OUT/SOURCE.sha256" > "$OUT/SOURCE.final.verify"
sha256sum -c "$OUT/BOOST_HEADERS.sha256" > "$OUT/BOOST_HEADERS.final.verify"
sha256sum -c "$OUT/COMPILER.sha256" > "$OUT/COMPILER.final.verify"
sha256sum -c "$OUT/EXECUTABLES.sha256" > "$OUT/EXECUTABLES.final.verify"
printf '%s\n' 'Scalar numeric + original checked compatibility only. Ordinary compiler region prologues are not certified allocation-free by this gate; worker admission separately requires source-bound final IR proof and region elision.' > "$OUT/SCOPE.txt"
echo NORMAL_SCALAR_NUMERIC_GATE_OK
