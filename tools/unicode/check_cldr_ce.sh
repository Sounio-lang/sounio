#!/usr/bin/env bash
# Pinned CLDR49 own CE compiler plus independent ICU79.1 oracle.
# Native Sounio locale conformance remains a separate mandatory gate.
set -euo pipefail
: "${SLURM_JOB_ID:?run under Slurm}"
: "${ICU_PREFIX:?isolated pinned ICU79.1 prefix}"
test "$#" -eq 1
OUT=$1
test ! -e "$OUT"
mkdir -p "$OUT/upstream"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export LD_LIBRARY_PATH="$ICU_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf '%s  %s\n' \
  1104a80925a97d6c1cb7a480ac5fa15650edf67dc7eff63c86a3cc065e801778 stdlib/text/data/cldr49-source.tar.gz \
  726d30c1398ef4dfaa1217faf0760521c052ed1cdf1c87908947e2d8e82ed362 stdlib/text/data/ucd18-source.tar.gz \
  0e7090aa13324cbf2c52e0e8195713df68a82f8e1fae58b5bf669a88f6550fa6 "$ICU_PREFIX/lib/libicuuc.so.79.1" \
  24e3bc88c7b9c588d7bfaab4f3803cbe0476e3e93e08de1863617b98603439bd "$ICU_PREFIX/lib/libicui18n.so.79.1" \
  2f26922e9964e3ca2da1e08fc9dde67feeb99b153a4a2b74ca7fc3ec13a0dba2 "$ICU_PREFIX/lib/libicudata.so.79.1" > "$OUT/PINNED_INPUTS.sha256"
sha256sum -c "$OUT/PINNED_INPUTS.sha256" > "$OUT/PINNED_INPUTS.before.verify"
sha256sum tools/unicode/{cldr_generate,cldr_registry,cldr_fractional,cldr_build,cldr_data_generate,cldr_graph_oracle}.cpp tools/unicode/check_cldr_ce.sh > "$OUT/SOURCE.sha256"
g++ --version > "$OUT/CXX_VERSION.txt"
sha256sum "$(command -v g++)" > "$OUT/CXX.sha256"
tar -xzf stdlib/text/data/cldr49-source.tar.gz -C "$OUT/upstream"
tar -xzf stdlib/text/data/ucd18-source.tar.gz -C "$OUT/upstream"
find "$OUT/upstream" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/UPSTREAM.sha256"
flags=(-std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation)
g++ "${flags[@]}" tools/unicode/cldr_fractional.cpp -o "$OUT/fractional"
"$OUT/fractional" "$OUT/upstream/selected/common/uca/FractionalUCA.txt" "$OUT/fractional.tsv" > "$OUT/fractional.stdout" 2> "$OUT/fractional.stderr"
g++ "${flags[@]}" tools/unicode/cldr_data_generate.cpp -o "$OUT/generate"
"$OUT/generate" "$OUT/upstream/selected" "$OUT/upstream/ucd-18.0.0/UnicodeData.txt" "$OUT/locales.bin" "$OUT/data.tsv" > "$OUT/generate.stdout" 2> "$OUT/generate.stderr"
"$OUT/generate" "$OUT/upstream/selected" "$OUT/upstream/ucd-18.0.0/UnicodeData.txt" "$OUT/repeat.bin" "$OUT/repeat.tsv" > "$OUT/repeat.stdout" 2> "$OUT/repeat.stderr"
cmp "$OUT/locales.bin" "$OUT/repeat.bin"
cmp "$OUT/data.tsv" "$OUT/repeat.tsv"
g++ "${flags[@]}" -I"$ICU_PREFIX/include" tools/unicode/cldr_graph_oracle.cpp -L"$ICU_PREFIX/lib" -licuuc -licui18n -o "$OUT/oracle"
"$OUT/oracle" "$OUT/upstream/selected" "$OUT/upstream/ucd-18.0.0/UnicodeData.txt" > "$OUT/oracle.stdout" 2> "$OUT/oracle.stderr"
g++ "${flags[@]}" -DSOUNIO_CLDR_WRONG_PRIMARY_ORACLE_CONTROL -I"$ICU_PREFIX/include" tools/unicode/cldr_graph_oracle.cpp -L"$ICU_PREFIX/lib" -licuuc -licui18n -o "$OUT/wrong-primary"
if "$OUT/wrong-primary" "$OUT/upstream/selected" "$OUT/upstream/ucd-18.0.0/UnicodeData.txt" aa > "$OUT/wrong-primary.stdout" 2> "$OUT/wrong-primary.stderr"; then
 echo "wrong primary implementation was accepted" >&2
 exit 91
fi
grep -F 'CLDR_CE_GRAPH_ORACLE_ERROR CE graph ordering' "$OUT/wrong-primary.stderr" > "$OUT/WRONG_PRIMARY.verify"
sha256sum -c "$OUT/PINNED_INPUTS.sha256" > "$OUT/PINNED_INPUTS.final.verify"
sha256sum -c "$OUT/SOURCE.sha256" > "$OUT/SOURCE.final.verify"
sha256sum -c "$OUT/UPSTREAM.sha256" > "$OUT/UPSTREAM.final.verify"
sha256sum -c "$OUT/CXX.sha256" > "$OUT/CXX.final.verify"
sha256sum "$OUT"/{fractional,generate,oracle,wrong-primary,locales.bin} > "$OUT/OUTPUTS.sha256"
echo CLDR_OWN_CE_DEFAULT_ORACLE_AND_DETERMINISM_OK
