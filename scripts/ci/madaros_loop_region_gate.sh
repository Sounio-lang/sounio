#!/usr/bin/env bash
# Loop regions (docs/decisions/adr-010-madaros-region-reclamation.md,
# amendment "Loop regions"): one region spans each eligible loop and is reset
# in place at every iteration end; loop-carried aggregates are copied out
# first (__rgn_reset / __rgn_settle / __rgn_copy_payload, ir/lower.sio
# LOWER_LOOP_RGN_*).
#
#  1. Builtin ids. The native dispatch is a flat `builtin_id == N` chain, so a
#     shared id silently runs the wrong emitter (#2966: __ptr_of_ref on 72 ran
#     array_concat). Every id in the chain must be distinct, and every region
#     builtin name must map to an id that the chain dispatches exactly once.
#  2. Reclaim. madaros_loop_region_reclaim.sio drives 10M handle-sized
#     temporaries through main's loops, carries aggregates with integer words
#     across 10M iterations (the case the numeric-word scan used to keep
#     forever) and checks a 513-word carried array element by element.
#  3. Teeth. With SOUNIO_NO_REGION_RECLAIM=1 the same witness must die on the
#     handle wall (exit 182): reclamation, not luck, is what makes it pass.
#  4. Mutation. madaros_loop_region_escape.sio prints a bitmask (case k sets
#     bit k-1) as LOOP_REGION_ESCAPE_MASK=<n> -- on stdout, because the mask
#     passes 255 and an exit status would drop bit 8 (case 9 alone exited 0
#     before this was changed). Removing one store barrier, or one rule of the
#     loop escape analysis (SOUNIO_LOOP_RGN_SABOTAGE), must flip exactly the
#     cases that rule protects; the plain build must print mask 0 and exit 0.
#  5. Merge veto. Lanes that lower modules into separate function tables and
#     merge them veto every region when a side may hold unhooked casts
#     (lower_rgn_merge_check). The reclaim witness with one import, in the
#     place+remap lane: 0 plain, 182 with SOUNIO_RGN_FORCE_MERGE_VETO=1 (the
#     veto really removes the regions); in the default lane, one shared table
#     and no such merge, 0 with the same switch.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="[madaros-loop-region]"
fail() { echo "$TAG FAIL: $*" >&2; exit 1; }

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "$TAG SKIP: Linux-only gate" >&2; exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *) echo "$TAG SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac

# ---------------------------------------------------------------- 1 builtin ids
CG="$ROOT_DIR/self-hosted/native/codegen_x86_linux.sio"
[[ -f "$CG" ]] || fail "missing $CG"
# The dispatch chain proper (other functions reuse the identifier locally).
chain="$(sed -n '/^fn native_v2_emit_builtin_by_id_into/,/^}/p' "$CG")"
[[ -n "$chain" ]] || fail "could not read native_v2_emit_builtin_by_id_into"
dups="$(grep -oE 'if builtin_id == [0-9]+ \{' <<<"$chain" | grep -oE '[0-9]+' | sort -n | uniq -d | tr '\n' ' ')"
[[ -z "$dups" ]] || fail "builtin ids dispatched more than once: $dups"
region_fn="$(sed -n '/^fn native_v2_region_builtin_id/,/^}/p' "$CG")"
region_ids="$(grep -oE 'name_eq_lit\(n, "__[a-z_]+"\) \{ return [0-9]+ \}' <<<"$region_fn" \
  | sed -E 's/.*"(__[a-z_]+)".*return ([0-9]+).*/\1 \2/')"
[[ -n "$region_ids" ]] || fail "could not read native_v2_region_builtin_id"
rdups="$(awk '{print $2}' <<<"$region_ids" | sort -n | uniq -d | tr '\n' ' ')"
[[ -z "$rdups" ]] || fail "region builtins share ids: $rdups"
# Name -> id maps elsewhere alias on purpose (malloc and heap_alloc are both
# 23), so names in general may share an id. Region ids may not: each must be
# dispatched exactly once and returned by no name test outside
# native_v2_region_builtin_id.
outside="$(sed '/^fn native_v2_region_builtin_id/,/^}/d' "$CG" | grep -E '^\s*if [^{]*name[^{]*\) \{ return [0-9]+ \}')"
while read -r name id; do
  n="$(grep -cE "if builtin_id == $id \{" <<<"$chain" || true)"
  [[ "$n" == 1 ]] || fail "$name -> id $id is dispatched $n times (want 1)"
  if grep -qE "\{ return $id \}" <<<"$outside"; then
    fail "$name's id $id is also returned for another name: $(grep -E "\{ return $id \}" <<<"$outside" | head -1 | sed 's/^ *//')"
  fi
done <<<"$region_ids"
for name in __rgn_reset __rgn_settle __rgn_copy_payload; do
  grep -qE "^$name " <<<"$region_ids" || fail "$name has no region builtin id"
done
echo "$TAG PASS: $(wc -l <<<"$region_ids") region builtins, every dispatched id distinct"

WORK="$(mktemp -d /tmp/sounio-madaros-loop-region.XXXXXX)"
if [[ "${SOUNIO_MADAROS_LOOP_REGION_GATE_KEEP:-0}" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${MADAROS_RAW_BIN:-}"
if [[ -z "$RAW" ]]; then
  RAW="$WORK/madaros-from-source.elf"
  echo "$TAG no MADAROS_RAW_BIN; building Madaros from source"
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" >"$WORK/build.log" 2>&1; then
    tail -n 40 "$WORK/build.log" >&2 || true
    fail "could not build Madaros from source"
  fi
fi
[[ -x "$RAW" ]] || fail "Madaros is missing or not executable: $RAW"
[[ "$(head -c 2 "$RAW")" != '#!' ]] || fail "not a raw ELF (a wrapper script?): $RAW"
case "$RAW" in /*) ;; *) RAW="$PWD/$RAW" ;; esac

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
unset SOUNIO_NO_REGION_RECLAIM SOUNIO_RGN_BARRIER_SABOTAGE SOUNIO_LOOP_RGN_SABOTAGE SOUNIO_LOOP_RGN_DIAG SOUNIO_RGN_FORCE_MERGE_VETO SOUNIO_MADAROS_DEP_MERGE

default_build() { MADAROS_RAW_BIN="$RAW" "$ROOT_DIR/bin/madaros" build "$1" -o "$2"; }
run_elf() { local rc=0; timeout 600 "$1" >"$2" 2>"$2.err" || rc=$?; echo "$rc"; }

RP="$ROOT_DIR/tests/run-pass"
RECLAIM="$RP/madaros_loop_region_reclaim.sio"
ESCAPE="$RP/madaros_loop_region_escape.sio"
[[ -f "$RECLAIM" && -f "$ESCAPE" ]] || fail "missing loop-region fixtures"

# ---------------------------------------------------------------- 2 reclaim
default_build "$RECLAIM" "$WORK/reclaim.elf" >"$WORK/reclaim.log" 2>&1 \
  || { tail -n 30 "$WORK/reclaim.log" >&2; fail "reclaim: compile failed"; }
rc="$(run_elf "$WORK/reclaim.elf" "$WORK/reclaim.out")"
[[ "$rc" == 0 ]] || { cat "$WORK/reclaim.out.err" >&2; fail "reclaim: exited $rc (case $rc failed, or 182 = handles full)"; }
grep -qxF LOOP_REGION_RECLAIM_OK "$WORK/reclaim.out" || fail "reclaim: marker missing"
echo "$TAG PASS: reclaim witness (10M temporaries, integer-carrying aggregates, 513-word content check)"

# ---------------------------------------------------------------- 3 teeth
SOUNIO_NO_REGION_RECLAIM=1 default_build "$RECLAIM" "$WORK/reclaim.optout.elf" >"$WORK/reclaim.optout.log" 2>&1 \
  || { tail -n 30 "$WORK/reclaim.optout.log" >&2; fail "opt-out: compile failed"; }
rc="$(run_elf "$WORK/reclaim.optout.elf" "$WORK/reclaim.optout.out")"
[[ "$rc" == 182 ]] || fail "opt-out: reclaim witness exited $rc, want 182 (it no longer needs reclamation: witness has no teeth)"
grep -qF 'madaros: handles full' "$WORK/reclaim.optout.out.err" || fail "opt-out: exit 182 without the handles-full diagnostic"
echo "$TAG PASS: without reclamation the witness dies on the handle wall (exit 182)"

# ---------------------------------------------------------------- 4 mutation
# <env assignment> <expected mask>. Mask bits: 1 field store, 2 index store,
# 4 ptr store, 8 outer local, 16 return from a loop, 32 break, 64 nested,
# 128 outer local of an uncopyable type (refused region; sabotage 1),
# 256 f64 array copied from a parameter and carried (the float-marker case).
# 2/3: under spanning regions the normal loop exit shares the break label, so
# making exits reset breaks every case that exits a loop holding a kept value.
MASKS=(
  "PLAIN=1 0"
  "SOUNIO_RGN_BARRIER_SABOTAGE=field 1"
  "SOUNIO_RGN_BARRIER_SABOTAGE=index 2"
  "SOUNIO_RGN_BARRIER_SABOTAGE=storeptr 4"
  "SOUNIO_LOOP_RGN_SABOTAGE=1 128"
  "SOUNIO_LOOP_RGN_SABOTAGE=2 376"
  "SOUNIO_LOOP_RGN_SABOTAGE=3 504"
  "SOUNIO_LOOP_RGN_SABOTAGE=4 328"
)
for m in "${MASKS[@]}"; do
  read -r assign want <<<"$m"
  label="${assign//=/-}"
  env "$assign" MADAROS_RAW_BIN="$RAW" "$ROOT_DIR/bin/madaros" build "$ESCAPE" -o "$WORK/esc.$label.elf" >"$WORK/esc.$label.log" 2>&1 \
    || { tail -n 30 "$WORK/esc.$label.log" >&2; fail "escape [$assign]: compile failed"; }
  rc="$(run_elf "$WORK/esc.$label.elf" "$WORK/esc.$label.out")"
  mask="$(sed -n 's/^LOOP_REGION_ESCAPE_MASK=\([0-9][0-9]*\)$/\1/p' "$WORK/esc.$label.out" | head -1)"
  [[ -n "$mask" ]] || fail "escape [$assign]: no LOOP_REGION_ESCAPE_MASK line (exit $rc)"
  if [[ "$mask" == 0 ]]; then want_rc=0; else want_rc=1; fi
  [[ "$rc" == "$want_rc" ]] || fail "escape [$assign]: mask $mask but exit $rc (want $want_rc)"
  [[ "$mask" == "$want" ]] || fail "escape [$assign]: mask $mask, want $want"
  echo "$TAG PASS: escape witness [$assign] mask $mask"
done
# ---------------------------------------------------------------- 5 merge veto
IMPORTED="$WORK/reclaim_imported.sio"
{ echo "use math::dd64::{dd_zero}"; grep -v '^//@' "$RECLAIM" | sed '/^fn main()/a\    let _z = dd_zero()'; } >"$IMPORTED"
grep -q 'let _z = dd_zero()' "$IMPORTED" || fail "merge veto: could not build the imported witness"
VETO_CASES=(
  "place SOUNIO_MADAROS_DEP_MERGE=place 0"
  "place_vetoed SOUNIO_MADAROS_DEP_MERGE=place,SOUNIO_RGN_FORCE_MERGE_VETO=1 182"
  "default_vetoed SOUNIO_RGN_FORCE_MERGE_VETO=1 0"
)
for vc in "${VETO_CASES[@]}"; do
  read -r name assigns want <<<"$vc"
  IFS=, read -r -a envs <<<"$assigns"
  env "${envs[@]}" MADAROS_RAW_BIN="$RAW" "$ROOT_DIR/bin/madaros" build "$IMPORTED" -o "$WORK/veto_$name.elf" >"$WORK/veto_$name.log" 2>&1 \
    || { tail -n 30 "$WORK/veto_$name.log" >&2; fail "merge veto [$name]: compile failed"; }
  rc="$(run_elf "$WORK/veto_$name.elf" "$WORK/veto_$name.out")"
  [[ "$rc" == "$want" ]] || fail "merge veto [$name]: exited $rc, want $want"
  if [[ "$want" == 182 ]]; then
    grep -q 'region veto' "$WORK/veto_$name.log" || fail "merge veto [$name]: the compile did not report the veto"
  fi
  echo "$TAG PASS: merge veto [$name] exit $rc"
done
echo "$TAG PASS"
