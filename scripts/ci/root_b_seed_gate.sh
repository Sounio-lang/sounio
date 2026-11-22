#!/usr/bin/env bash
# Root B, seed loader (bootstrap/roots/b), and the cross-check between the
# two roots' loaders. seed-b is written by hand separately from seed-a, in
# its own format (octal groups, ';' comments).
#
#   1. self-reproduction: seed-b < seed-b.oct is byte-identical to seed-b;
#   2. cross-check: each loader assembles the other's listing, written in
#      its own format, into the other's committed bytes:
#        seed-a < roots/b/seed-b.hex == seed-b
#        seed-b < roots/a/seed-a.oct == seed-a
#   3. fixed vectors: accepted inputs and their bytes, refused inputs (exit 1);
#   4. all 256 byte values round-trip;
#   5. differential fuzz against the C++23 reference model seed_b_ref.cpp.
#
# The reference model is a test oracle only; nothing it produces is committed
# or used on the bootstrap path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
A="$ROOT/bootstrap/roots/a"
B="$ROOT/bootstrap/roots/b"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
CXX_BIN="${CXX:-g++}"
FUZZ_N="${ROOT_B_FUZZ_N:-5000}"
FUZZ_SEED="${ROOT_B_FUZZ_SEED:-20261115}"

echo "== root_b_seed_gate =="
cp "$A/seed-a" "$OUT/seed-a"; cp "$B/seed-b" "$OUT/seed-b"; chmod +x "$OUT/seed-a" "$OUT/seed-b"
"$CXX_BIN" -std=c++23 -O2 -o "$OUT/ref" "$B/seed_b_ref.cpp"
fail=0

# 1 and 2: self-reproduction and the cross-check
assemble() {  # name loader listing expected
  local rc=0
  "$OUT/$2" < "$3" > "$OUT/$1.out" || rc=$?
  if [[ "$rc" -eq 0 ]] && cmp -s "$OUT/$1.out" "$4"; then
    echo "ok   $1 ($(wc -c < "$4") bytes)"
  else
    echo "FAIL: $1: rc=$rc"; cmp "$OUT/$1.out" "$4" || true; fail=1
  fi
}
assemble self-reproduction-b  seed-b "$B/seed-b.oct" "$B/seed-b"
assemble cross-a-assembles-b  seed-a "$B/seed-b.hex" "$B/seed-b"
assemble cross-b-assembles-a  seed-b "$A/seed-a.oct" "$A/seed-a"

# 3. fixed vectors: name, input (printf format), expected rc, expected output (printf format)
vec() {
  local name="$1" in="$2" want_rc="$3" want_out="$4" rc=0
  printf "$in" > "$OUT/v.in"; printf "$want_out" > "$OUT/v.want"
  "$OUT/seed-b" < "$OUT/v.in" > "$OUT/v.out" || rc=$?
  if [[ "$rc" -ne "$want_rc" ]]; then
    echo "FAIL: $name: rc=$rc, want $want_rc"; fail=1
  elif [[ "$want_rc" -eq 0 ]] && ! cmp -s "$OUT/v.out" "$OUT/v.want"; then
    echo "FAIL: $name: wrong bytes"; fail=1
  else
    echo "ok   $name"
  fi
}
vec empty                 ''                        0 ''
vec group                 '101'                     0 'A'
vec groups_ws             '101 102\t103\n'          0 'ABC'
vec max_group             '377'                     0 '\377'
vec comment               '; x\n101 ; y\n102'       0 'AB'
vec comment_at_eof        '101 ; no newline'        0 'A'
vec comment_any_bytes     ';\001\377\r\n172'        0 'z'
vec space_in_group        '1 01'                    1 ''
vec comment_in_group      '10;1\n'                  1 ''
vec first_digit_4         '401'                     1 ''
vec digit_8               '108'                     1 ''
vec hex_letter            '10a'                     1 ''
vec cr_refused            '101\r\n'                 1 ''
vec incomplete_group      '1011'                    1 ''
vec hash_refused          '# not a comment here'    1 ''
vec nul_refused           '\000'                    1 ''

# 4. all 256 byte values
"$OUT/ref" all256 > "$OUT/all.oct"
rc=0; "$OUT/seed-b" < "$OUT/all.oct" > "$OUT/all.out" || rc=$?
"$OUT/ref" decode < "$OUT/all.oct" > "$OUT/all.want"
if [[ "$rc" -eq 0 ]] && [[ "$(wc -c < "$OUT/all.out")" -eq 256 ]] && cmp -s "$OUT/all.out" "$OUT/all.want"; then
  echo "ok   all 256 byte values"
else
  echo "FAIL: all 256 byte values: rc=$rc"; fail=1
fi

# 5. differential fuzz
mkdir -p "$OUT/fz"
"$OUT/ref" fuzz "$FUZZ_SEED" "$FUZZ_N" "$OUT/fz"
bad=0; acc=0; rej=0
for ((i = 0; i < FUZZ_N; i++)); do
  rc=0; "$OUT/seed-b" < "$OUT/fz/$i.in" > "$OUT/fz/$i.got" || rc=$?
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

if [[ $fail -ne 0 ]]; then echo "ROOT_B_SEED_GATE=FAIL"; exit 1; fi
echo "ROOT_B_SEED_GATE=PASS"
