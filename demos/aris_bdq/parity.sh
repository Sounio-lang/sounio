#!/usr/bin/env bash
# Parity gate: Sounio (demos/aris_bdq/bdq_lung.sio) vs the independent C++23 oracle
# (demos/aris_bdq/ref/bdq.cpp). Every BDQ| line must agree to a relative 1e-5, and the
# Sounio program must certify its own mass ledgers.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

g++ -std=c++23 -O3 -o "$work/bdq" "$here/ref/bdq.cpp"
"$work/bdq" parity > "$work/oracle.txt"

(cd "$root" && ./bin/souc run demos/aris_bdq/bdq_lung.sio) > "$work/sounio.log" 2>&1
grep -E '^(BDQ|GUM)\|' "$work/sounio.log" | grep -v ledger > "$work/sounio.txt"
grep -q '^BDQ_LUNG_LEDGERS_PASS$' "$work/sounio.log" || { echo "FAIL: Sounio ledgers"; exit 1; }

fail=0
while IFS='=' read -r key ref; do
  got="$(grep -F "${key}=" "$work/sounio.txt" | head -1 | cut -d= -f2)"
  if [ -z "$got" ]; then echo "MISSING $key"; fail=1; continue; fi
  ok="$(awk -v a="$got" -v b="$ref" 'BEGIN{d=a-b; if(d<0)d=-d; s=(b<0?-b:b); if(s<1e-12)s=1e-12; print (d/s<=1e-5)?"ok":"bad"}')"
  printf '%-30s oracle=%-16s sounio=%-16s %s\n' "$key" "$ref" "$got" "$ok"
  [ "$ok" = ok ] || fail=1
done < "$work/oracle.txt"

if [ "$fail" -eq 0 ]; then echo "BDQ_PARITY_PASS"; else echo "BDQ_PARITY_FAIL"; exit 1; fi
