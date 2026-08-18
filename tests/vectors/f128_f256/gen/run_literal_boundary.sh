#!/usr/bin/env bash
# Rebuild + regenerate literal-boundary / double-rounding vectors (Wave 3 WS-G).
set -euo pipefail
cd "$(dirname "$0")"

gcc -O2 -Wall -Wextra -Wno-unused-function \
  -o literal_boundary_gen literal_boundary_gen.c -lmpfr -lgmp

echo "MPFR: $(pkg-config --modversion mpfr 2>/dev/null || true)"
echo "GCC:  $(gcc --version | head -1)"

./literal_boundary_gen 2>literal_boundary_gen.stderr | tee /tmp/lit_all.jsonl >/dev/null
# split by format
python3 - <<'PY'
import json
n128 = n256 = 0
with open("/tmp/lit_all.jsonl") as inp, \
     open("../literal_boundary_f128.jsonl", "w") as f128, \
     open("../literal_boundary_f256.jsonl", "w") as f256:
    for line in inp:
        o = json.loads(line)
        if o["format"] == "binary128":
            f128.write(line)
            n128 += 1
        else:
            f256.write(line)
            n256 += 1
print(f"wrote literal_boundary_f128.jsonl lines={n128}")
print(f"wrote literal_boundary_f256.jsonl lines={n256}")
PY

echo "--- hashes ---"
md5sum ../literal_boundary_f128.jsonl ../literal_boundary_f256.jsonl
sha256sum ../literal_boundary_f128.jsonl ../literal_boundary_f256.jsonl
cat literal_boundary_gen.stderr
