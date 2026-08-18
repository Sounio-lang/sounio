#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gcc -O2 -Wall -Wextra -Wno-format -o wire_encoding_gen wire_encoding_gen.c
./wire_encoding_gen 2>wire_encoding_gen.stderr | tee /tmp/wire_all.jsonl >/dev/null
python3 - <<'PY'
import json, hashlib
n128 = n256 = 0
with open("/tmp/wire_all.jsonl") as inp, \
     open("../wire_f128.jsonl", "w") as f128, \
     open("../wire_f256.jsonl", "w") as f256:
    for line in inp:
        o = json.loads(line)
        if o["format"] == "binary128":
            f128.write(line); n128 += 1
        else:
            f256.write(line); n256 += 1
print(f"wire_f128.jsonl lines={n128}")
print(f"wire_f256.jsonl lines={n256}")
for p in ("../wire_f128.jsonl", "../wire_f256.jsonl"):
    data = open(p, "rb").read()
    print(p, "md5", hashlib.md5(data).hexdigest())
    print(p, "sha256", hashlib.sha256(data).hexdigest())
PY
cat wire_encoding_gen.stderr
