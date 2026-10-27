#!/usr/bin/env bash
# Strict CLDR49 AST parser and ICU79.1 oracle preparation. No production weights.
set -euo pipefail
: "${SLURM_JOB_ID:?run under Slurm}"
: "${ICU_PREFIX:?isolated ICU79.1 oracle prefix}"
test "$#" -eq 1
OUT=$1
test ! -e "$OUT"
mkdir -p "$OUT/source"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export LD_LIBRARY_PATH="$ICU_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sha256sum tools/unicode/cldr_generate.cpp tools/unicode/cldr_oracle.cpp stdlib/text/data/cldr49-source.tar.gz "$ICU_PREFIX"/lib/libicu{uc,i18n,data}.so.79.1 > "$OUT/INPUTS.sha256"
tar -xzf stdlib/text/data/cldr49-source.tar.gz -C "$OUT/source"
find "$OUT/source" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCES.sha256"
g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation tools/unicode/cldr_generate.cpp -o "$OUT/generate"
"$OUT/generate" "$OUT/source/selected" "$OUT/inventory.tsv"
g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation -I"$ICU_PREFIX/include" tools/unicode/cldr_oracle.cpp -L"$ICU_PREFIX/lib" -licuuc -licui18n -o "$OUT/oracle"
(cd "$OUT" && ./oracle source/selected)
sha256sum -c "$OUT/INPUTS.sha256" > "$OUT/INPUTS.final.verify"
sha256sum -c "$OUT/SOURCES.sha256" > "$OUT/SOURCES.final.verify"
echo CLDR_AST_GATE_OK
