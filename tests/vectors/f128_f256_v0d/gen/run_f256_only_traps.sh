#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gcc -O2 -Wall -Wextra -Wno-unused-function -o f256_only_traps_gen f256_only_traps_gen.c -lmpfr -lgmp -lm
./f256_only_traps_gen 2>f256_only_traps_gen.stderr | tee ../f256_only_traps.jsonl >/dev/null
md5sum ../f256_only_traps.jsonl
sha256sum ../f256_only_traps.jsonl
cat f256_only_traps_gen.stderr
