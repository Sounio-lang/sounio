#!/usr/bin/env bash
# C++23-only UCA18 data gate; run under Slurm with isolated ICU79.1 oracle prefix.
set -euo pipefail
: "${SLURM_JOB_ID:?run this gate under Slurm}"
: "${ICU_PREFIX:?set isolated ICU79.1 oracle prefix}"
test "$#" -eq 1
OUT=$1
test ! -e "$OUT"
mkdir -p "$OUT/source"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export LD_LIBRARY_PATH="$ICU_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT"
sha256sum tools/unicode/uca_generate.cpp tools/unicode/unicode_generate.cpp tests/reference/wave3_uca_reference.cpp stdlib/text/data/uca18-source.tar.gz stdlib/text/data/uca18-cldr49.bin > "$OUT/INPUTS.sha256"
sha256sum "$ICU_PREFIX"/lib/libicu{uc,i18n,data}.so.79.1 >> "$OUT/INPUTS.sha256"
tar -xzf stdlib/text/data/uca18-source.tar.gz -C "$OUT/source"
(cd "$OUT/source" && sha256sum -c SOURCES.sha256) > "$OUT/SOURCES.verify"
g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation tools/unicode/uca_generate.cpp -o "$OUT/generate"
"$OUT/generate" "$OUT/source/allkeys.txt" "$OUT/source/common/uca/allkeys_CLDR.txt" "$OUT/source/common/dtd/ldml.dtd" "$OUT/uca.bin" "$OUT/report.tsv"
"$OUT/generate" "$OUT/source/allkeys.txt" "$OUT/source/common/uca/allkeys_CLDR.txt" "$OUT/source/common/dtd/ldml.dtd" "$OUT/repeat.bin" "$OUT/repeat.tsv"
cmp "$OUT/uca.bin" "$OUT/repeat.bin"
cmp "$OUT/report.tsv" "$OUT/repeat.tsv"
cmp "$OUT/uca.bin" stdlib/text/data/uca18-cldr49.bin
g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation -I"$ICU_PREFIX/include" tests/reference/wave3_uca_reference.cpp -L"$ICU_PREFIX/lib" -licuuc -licui18n -o "$OUT/reference"
(cd "$OUT" && ./reference source/allkeys.txt source/common/uca/allkeys_CLDR.txt uca.bin source/CollationTest/CollationTest_NON_IGNORABLE.txt source/CollationTest/CollationTest_SHIFTED.txt)
sha256sum -c "$OUT/INPUTS.sha256" > "$OUT/INPUTS.final.verify"
(cd "$OUT/source" && sha256sum -c SOURCES.sha256) > "$OUT/SOURCES.final.verify"
sha256sum "$OUT"/{generate,reference,uca.bin,report.tsv} > "$OUT/OUTPUTS.sha256"
echo UCA_DATA_GATE_OK

