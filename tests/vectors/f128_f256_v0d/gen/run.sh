#!/usr/bin/env bash
# Rebuild + regenerate V0-D arithmetic hard-case corpora.
set -euo pipefail
cd "$(dirname "$0")"

gcc -O2 -Wall -Wextra -Wno-unused-function \
  -o arith_hard_gen arith_hard_gen.c -lmpfr -lgmp -lm

echo "MPFR: $(pkg-config --modversion mpfr 2>/dev/null || true)"
echo "GCC:  $(gcc --version | head -1)"

./arith_hard_gen 2>arith_hard_gen.stderr | tee /tmp/arith_hard_all.jsonl >/dev/null

python3 - <<'PY'
import json, hashlib
n128 = n256 = 0
with open("/tmp/arith_hard_all.jsonl") as inp, \
     open("../arith_hard_f128.jsonl", "w") as f128, \
     open("../arith_hard_f256.jsonl", "w") as f256:
    for line in inp:
        o = json.loads(line)
        if o["format"] == "binary128":
            f128.write(line)
            n128 += 1
        else:
            f256.write(line)
            n256 += 1
print(f"wrote arith_hard_f128.jsonl lines={n128}")
print(f"wrote arith_hard_f256.jsonl lines={n256}")
for p in ("../arith_hard_f128.jsonl", "../arith_hard_f256.jsonl"):
    data = open(p, "rb").read()
    print(p, "md5", hashlib.md5(data).hexdigest())
    print(p, "sha256", hashlib.sha256(data).hexdigest())
PY

cat arith_hard_gen.stderr
