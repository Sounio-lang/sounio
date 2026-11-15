#!/usr/bin/env bash
# madaros_field_name_hash_refusal_witness_gate.sh — ADR-026 amendment (2026-11-09)
#
# Executable witness for the Stage 0 refusal, run against a current-source
# Madaros (MADAROS_RAW_BIN). field_name_hash_refused_gate.sh only measures the
# source; this one compiles programs.
#
# Negative witnesses: `codec` has a different index in A, B and C, and the
# receiver is an element of a local array literal or of a local tuple, whose
# struct type the lowerer does not know. Measured 2026-11-09 with the
# committed prebuilt Madaros v0.80.0 (name-hash fallback): both build, and both
# print 0 where the program says 30 and 20 — the hash read an unrelated slot.
# A current-source compiler must refuse each one: nonzero exit, the reason-10
# diagnostic naming `.codec` in `main`, and no output ELF.
#
# Positive witness: tests/run-pass/field_name_collision_typed_receivers.sio,
# whose receivers are typed, must still print 20 7 7 5 20.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "madaros_field_name_hash_refusal_witness_gate"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true

SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-}}"
[[ -x "$SOUC" ]] || gate_fail "souc is not executable: $SOUC"
[[ -n "$RAW" && -x "$RAW" ]] || gate_fail "MADAROS_RAW_BIN must name a current-source Madaros ELF"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/field-hash-witness.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

write_case() {
  cat >"$WORK/$1.sio" <<SIO
struct A { x: i64, codec: i64 }
struct B { codec: i64, y: i64, z: i64 }
struct C { w: i64, v: i64, codec: i64 }

fn main() -> i64 with IO, Mut, Panic, Div, Alloc {
    $2
    0
}
SIO
}

write_case array_elem 'let arr = [A { x: 1, codec: 20 }, A { x: 2, codec: 30 }]
    println(arr[1].codec)'
write_case tuple_elem 'let t = (A { x: 1, codec: 20 }, 5)
    println(t.0.codec)'

for c in array_elem tuple_elem; do
  rm -f "$WORK/$c.elf"
  set +e
  ( ulimit -s unlimited 2>/dev/null || true
    MADAROS_RAW_BIN="$RAW" "$SOUC" compile "$WORK/$c.sio" -o "$WORK/$c.elf" ) >"$WORK/$c.log" 2>&1
  rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    sed -n '1,30p' "$WORK/$c.log" >&2
    gate_fail "$c: compile succeeded; a field index invented from the name was accepted"
  fi
  grep -q "field access(es) whose index is not in the layout table" "$WORK/$c.log" \
    || { sed -n '1,40p' "$WORK/$c.log" >&2; gate_fail "$c: refused (rc=$rc) but not with the reason-10 diagnostic"; }
  grep -qF -e '- `.codec` in function `main`' "$WORK/$c.log" \
    || { sed -n '1,40p' "$WORK/$c.log" >&2; gate_fail "$c: the diagnostic does not name \`.codec\` in \`main\`"; }
  [[ ! -s "$WORK/$c.elf" ]] || gate_fail "$c: an output ELF was written despite the refusal"
  echo "$c: refused rc=$rc, reason 10 names .codec in main, no ELF"
done

POS="$ROOT/tests/run-pass/field_name_collision_typed_receivers.sio"
set +e
( ulimit -s unlimited 2>/dev/null || true
  MADAROS_RAW_BIN="$RAW" "$SOUC" run "$POS" ) >"$WORK/pos.log" 2>&1
prc=$?
set -e
[[ "$prc" -eq 0 ]] || { sed -n '1,30p' "$WORK/pos.log" >&2; gate_fail "positive witness exited $prc"; }
want="20 7 7 5 20"
got="$(grep -E '^-?[0-9]+$' "$WORK/pos.log" | tr '\n' ' ' | sed 's/ $//')"
[[ "$got" == "$want" ]] || gate_fail "positive witness printed '$got', want '$want'"

gate_pass "array/tuple-element collisions refused with reason 10 and no ELF; typed collisions print $want"
echo "measured=yes"
exit 0
