#!/usr/bin/env bash
# Differential test: the hand-written stage-1 Sporos checker (hex, run in
# the X86Int model by x86run's file-feeding variant) against the Lean
# reference's verdict on the same file.
#   bootstrap/model/sporos_stage1_diff.sh [N]
# Stage 1 is check-only: it exits 0 exactly when decode and checkProg
# accept. The reference runs the program after accepting, so the
# comparison is by its refusal line, not its exit status: a file the
# reference refuses ("not a Sporos bytecode file", "refused by the
# checker") must make the hex exit 1; any other file must make it exit 0.
# Vectors: the committed hand-made vectors in
# bootstrap/roots/sporos/vectors (each also truncated and byte-flipped
# here), the example programs, N sporos_gen programs, and two
# corruptions of each generated program.
# The RunHex variant is generated from RunHex.lean in a scratch dir: the
# committed x86run feeds only a patterned file of a given size.
set -euo pipefail
N="${1:-20}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
HEX="$ROOT/bootstrap/roots/sporos/sporos-stage1.hex"
( cd "$ROOT/formal" && lake build sporos sporos-examples x86run >/dev/null )
REF="$ROOT/formal/.lake/build/bin/sporos"
g++ -std=c++23 -O2 -o "$W/gen" "$ROOT/bootstrap/model/sporos_gen.cpp"
"$ROOT/formal/.lake/build/bin/sporos-examples" "$W/ex" >/dev/null
awk '{ if ($0 ~ /path :: size :: rest/) sub(/size/, "fpath");
       if ($0 ~ /file of \{size\}/) sub(/\{size\}/, "{fpath}");
       if ($0 ~ /startState code \(pattern/) { print "      let fb ← IO.FS.readBinFile fpath"; sub(/\(pattern size.toNat!\)/, "(fb.toList.map UInt8.toNat)") }
       print }' \
    "$ROOT/formal/Bootstrap/RunHex.lean" > "$W/RunHexFile.lean"
grep -q readBinFile "$W/RunHexFile.lean" || { echo "RunHex.lean changed shape; update the awk above"; exit 1; }
cd "$ROOT/formal"; export LEAN_PATH="$PWD/.lake/build/lib/lean"
pass=0; fail=0
check() {
  local f="$1" err want got
  err=$("$REF" "$f" </dev/null 2>&1 >/dev/null || true)
  case "$err" in *refused*|*"not a Sporos"*) want=1;; *) want=0;; esac
  got=$(lean --run "$W/RunHexFile.lean" "$HEX" "$f" 2>&1 | sed -n '2p' || true)
  case "$got" in
    "exit $want,"*) pass=$((pass+1));;
    *) fail=$((fail+1)); echo "MISMATCH $f want=exit $want got=$got";;
  esac
}
for f in "$W"/ex/*.spb "$ROOT"/bootstrap/roots/sporos/vectors/*.spb; do
  check "$f"
  sz=$(wc -c < "$f"); [ "$sz" -gt 1 ] || continue
  head -c $((sz-1)) "$f" > "$W/v.trunc.spb"; check "$W/v.trunc.spb"
  cp "$f" "$W/v.flip.spb"
  printf '\xff' | dd of="$W/v.flip.spb" bs=1 seek=20 conv=notrunc 2>/dev/null; check "$W/v.flip.spb"
done
for i in $(seq 1 "$N"); do
  "$W/gen" "$i" > "$W/g$i.spb"; check "$W/g$i.spb"
  sz=$(wc -c < "$W/g$i.spb")
  head -c $((sz-1)) "$W/g$i.spb" > "$W/g$i.trunc.spb"; check "$W/g$i.trunc.spb"
  cp "$W/g$i.spb" "$W/g$i.flip.spb"
  printf '\xff' | dd of="$W/g$i.flip.spb" bs=1 seek=20 conv=notrunc 2>/dev/null; check "$W/g$i.flip.spb"
done
# On Linux x86_64 the same vectors also run as a real binary: assembled
# by seed-a itself (bootstrap/roots/a), executed by the kernel, and
# compared against sporos_fast. Elsewhere this phase is skipped and the
# model run above is the gate.
if [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ]; then
  A="$ROOT/bootstrap/roots/a"
  cp "$A/seed-a" "$W/seed-a"; chmod +x "$W/seed-a"
  "$W/seed-a" < "$A/seed-a.hex" > "$W/self" && cmp -s "$W/self" "$A/seed-a" \
    || { echo "FAIL: seed-a self-reproduction"; fail=$((fail+1)); }
  g++ -std=c++23 -O2 -o "$W/fast" "$ROOT/bootstrap/model/sporos_fast.cpp"
  "$W/seed-a" < "$HEX" > "$W/stage1"; chmod +x "$W/stage1"
  kpass=0; kfail=0
  kcheck() {
    local f="$1" err want got
    err=$("$W/fast" "$f" </dev/null 2>&1 >/dev/null || true)
    case "$err" in *refused*|*"not a Sporos"*) want=1;; *) want=0;; esac
    got=0; "$W/stage1" "$f" >/dev/null 2>&1 || got=$?
    if [ "$got" = "$want" ]; then kpass=$((kpass+1)); else kfail=$((kfail+1)); echo "KERNEL MISMATCH $f want=$want got=$got"; fi
  }
  for f in "$W"/ex/*.spb "$ROOT"/bootstrap/roots/sporos/vectors/*.spb; do
    kcheck "$f"
    sz=$(wc -c < "$f"); [ "$sz" -gt 1 ] || continue
    head -c $((sz-1)) "$f" > "$W/v.trunc.spb"; kcheck "$W/v.trunc.spb"
    cp "$f" "$W/v.flip.spb"
    printf '\xff' | dd of="$W/v.flip.spb" bs=1 seek=20 conv=notrunc 2>/dev/null; kcheck "$W/v.flip.spb"
  done
  for i in $(seq 1 "$N"); do kcheck "$W/g$i.spb"; kcheck "$W/g$i.trunc.spb"; kcheck "$W/g$i.flip.spb"; done
  echo "sporos_stage1_diff (kernel via seed-a): pass=$kpass fail=$kfail"
  fail=$((fail+kfail))
fi
echo "sporos_stage1_diff: pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
