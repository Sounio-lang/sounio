#!/usr/bin/env bash
# Root A, seed loader (bootstrap/roots/a): the hand-written seed-a binary
# must be exactly what its own annotated listing describes, and must decode
# format A as specified, refusing everything else.
#
#   1. self-reproduction: seed-a < seed-a.hex is byte-identical to seed-a;
#   2. fixed vectors: accepted inputs and their bytes, refused inputs (exit 1);
#   3. all 256 byte values round-trip;
#   4. differential fuzz against the C++23 reference model seed_a_ref.cpp:
#      the same exit status and the same bytes written, on random valid,
#      commented and arbitrary inputs.
#
# The reference model is a test oracle only; nothing it produces is committed
# or used on the bootstrap path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
A="$ROOT/bootstrap/roots/a"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
CXX_BIN="${CXX:-g++}"
FUZZ_N="${ROOT_A_FUZZ_N:-5000}"
FUZZ_SEED="${ROOT_A_FUZZ_SEED:-20261114}"

echo "== root_a_seed_gate =="
cp "$A/seed-a" "$OUT/seed-a"; chmod +x "$OUT/seed-a"
"$CXX_BIN" -std=c++23 -O2 -o "$OUT/ref" "$A/seed_a_ref.cpp"
fail=0

# 1. self-reproduction
rc=0; "$OUT/seed-a" < "$A/seed-a.hex" > "$OUT/self" || rc=$?
if [[ "$rc" -eq 0 ]] && cmp -s "$OUT/self" "$A/seed-a"; then
  echo "ok   self-reproduction ($(wc -c < "$A/seed-a") bytes)"
else
  echo "FAIL: self-reproduction: rc=$rc"; cmp "$OUT/self" "$A/seed-a" || true; fail=1
fi

# 2. fixed vectors: name, input (printf format), expected rc, expected output (printf format)
vec() {
  local name="$1" in="$2" want_rc="$3" want_out="$4" rc=0
  printf "$in" > "$OUT/v.in"; printf "$want_out" > "$OUT/v.want"
  "$OUT/seed-a" < "$OUT/v.in" > "$OUT/v.out" || rc=$?
  if [[ "$rc" -ne "$want_rc" ]]; then
    echo "FAIL: $name: rc=$rc, want $want_rc"; fail=1
  elif [[ "$want_rc" -eq 0 ]] && ! cmp -s "$OUT/v.out" "$OUT/v.want"; then
    echo "FAIL: $name: wrong bytes"; fail=1
  else
    echo "ok   $name"
  fi
}
vec empty                ''                     0 ''
vec pair                 '41'                   0 'A'
vec pairs_ws             '41 42\t43\n'          0 'ABC'
vec split_digits         '4 1'                  0 'A'
vec comment              '# x\n41 # y\n42'      0 'AB'
vec comment_at_eof       '41 # no newline'      0 'A'
vec comment_any_bytes    '#\x01\xff\r\n7a'      0 'z'
vec upper_refused        '4A'                   1 ''
vec cr_refused           '41\r\n'               1 ''
vec odd_refused          '414'                  1 ''
vec letter_g_refused     '4g'                   1 ''
vec high_byte_refused    '\x80'                 1 ''
vec nul_refused          '\x00'                 1 ''

# 3. all 256 byte values
"$OUT/ref" all256 > "$OUT/all.hex"
rc=0; "$OUT/seed-a" < "$OUT/all.hex" > "$OUT/all.out" || rc=$?
"$OUT/ref" decode < "$OUT/all.hex" > "$OUT/all.want"
if [[ "$rc" -eq 0 ]] && [[ "$(wc -c < "$OUT/all.out")" -eq 256 ]] && cmp -s "$OUT/all.out" "$OUT/all.want"; then
  echo "ok   all 256 byte values"
else
  echo "FAIL: all 256 byte values: rc=$rc"; fail=1
fi

# 4. differential fuzz
mkdir -p "$OUT/fz"
"$OUT/ref" fuzz "$FUZZ_SEED" "$FUZZ_N" "$OUT/fz"
bad=0; acc=0; rej=0
for ((i = 0; i < FUZZ_N; i++)); do
  rc=0; "$OUT/seed-a" < "$OUT/fz/$i.in" > "$OUT/fz/$i.got" || rc=$?
  want_rc=$(cat "$OUT/fz/$i.rc")
  if [[ "$rc" -ne "$want_rc" ]] || ! cmp -s "$OUT/fz/$i.got" "$OUT/fz/$i.out"; then
    [[ $bad -lt 5 ]] && echo "FAIL: fuzz case $i: rc=$rc want $want_rc"
    bad=$((bad + 1))
  fi
  if [[ "$want_rc" -eq 0 ]]; then acc=$((acc + 1)); else rej=$((rej + 1)); fi
done
if [[ $bad -eq 0 ]]; then
  echo "ok   differential fuzz: $FUZZ_N cases ($acc accepted, $rej refused), seed $FUZZ_SEED"
else
  echo "FAIL: differential fuzz: $bad of $FUZZ_N cases differ"; fail=1
fi

if [[ $fail -ne 0 ]]; then echo "ROOT_A_SEED_GATE=FAIL"; exit 1; fi
echo "ROOT_A_SEED_GATE=PASS"
