#!/usr/bin/env bash
# Regenerate the f128 / f256 MPFR test-vector corpora deterministically.
#
# Usage: ./run.sh [<output-dir>]
#
# - output-dir defaults to the parent of this script (tests/vectors/f128_f256/)
# - rebuilds the generator if missing
# - prints tool versions, seeds, hashes to stderr so they can be captured
#   into GENERATION_RECEIPT.md
#
# Determinism: this script is deterministic on a given platform — the same
# GCC, MPFR, GMP versions and the same fixed PCG seed produce byte-identical
# outputs. Cross-platform exact reproduction is not guaranteed (different
# GMP/MPFR builds may differ in last-bit rounding for some edge inputs), so
# re-validate the receipt's hashes against the output of this script after
# any toolchain change.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$(dirname "$SCRIPT_DIR")}"
GEN_SRC="$SCRIPT_DIR/mpfr_vector_gen.c"
GEN_BIN="$SCRIPT_DIR/mpfr_vector_gen"

if [[ ! -f "$GEN_SRC" ]]; then
    echo "FATAL: $GEN_SRC not found" >&2
    exit 2
fi

# Build if missing or if source is newer than binary
if [[ ! -x "$GEN_BIN" ]] || [[ "$GEN_SRC" -nt "$GEN_BIN" ]]; then
    echo "[run.sh] building generator: $GEN_SRC" >&2
    gcc -O2 -Wall -Wextra -Wno-unused-function -Wno-unused-parameter \
        -Wno-shift-count-overflow \
        -o "$GEN_BIN" "$GEN_SRC" -lmpfr -lgmp
fi

mkdir -p "$OUT_DIR"
F128_OUT="$OUT_DIR/f128.jsonl"
F256_OUT="$OUT_DIR/f256.jsonl"

echo "[run.sh] generating f128 -> $F128_OUT" >&2
"$GEN_BIN" f128 "$F128_OUT"

echo "[run.sh] generating f256 -> $F256_OUT" >&2
"$GEN_BIN" f256 "$F256_OUT"

F128_LINES=$(wc -l < "$F128_OUT")
F256_LINES=$(wc -l < "$F256_OUT")
F128_MD5=$(md5sum "$F128_OUT" | awk '{print $1}')
F256_MD5=$(md5sum "$F256_OUT" | awk '{print $1}')
F128_SHA=$(sha256sum "$F128_OUT" | awk '{print $1}')
F256_SHA=$(sha256sum "$F256_OUT" | awk '{print $1}')

cat <<RECEIPT
[run.sh] generation complete
  f128: $F128_LINES lines, md5=$F128_MD5, sha256=$F128_SHA
  f256: $F256_LINES lines, md5=$F256_MD5, sha256=$F256_SHA

  pcg_state = 0x853c49e6748fea9b
  pcg_inc   = 0xda3e39cb94b95bdb
  gcc       = $(gcc --version | head -1)
  mpfr      = $(dpkg -s libmpfr-dev 2>/dev/null | awk '/^Version:/ {print $2}')
  gmp       = $(dpkg -s libgmp-dev  2>/dev/null | awk '/^Version:/ {print $2}')
  build cmd = gcc -O2 -Wall -Wextra -Wno-unused-function -Wno-unused-parameter -Wno-shift-count-overflow -o mpfr_vector_gen mpfr_vector_gen.c -lmpfr -lgmp
RECEIPT