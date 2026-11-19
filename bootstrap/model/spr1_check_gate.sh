#!/usr/bin/env bash
# Gate for spr1_check, the static x86-64 checker of Sporos bytecode:
# `spr1_check < FILE.spb` exits 0 when `Sporos.decode bs = some pr` and
# `Sporos.checkProg pr = true` (formal/Sporos/Bytecode.lean, Check.lean), 5 when
# the file is malformed or ill-typed, 6 on ENOMEM or a stack height of 2^48 or
# more, 3 when its three redundant verdicts all disagree.
#
#   bootstrap/model/spr1_check_gate.sh [N_FUZZ] [INJECT]
#
#   N_FUZZ  differential cases (default 10000)
#   INJECT  1: also run the silicon single-fault injection (Linux x86-64)
#
# Steps:
#   1. `bootstrap/roots/a/seed-a < bootstrap/model/spr1_check.hex` is
#      byte-identical to bootstrap/model/spr1_check.
#   2. The four examples, written three ways (the Lean oracle's `gen`, the
#      `sporos-examples` executable, the C++ fuzzer) and compared byte for
#      byte, exit 0; the fixed vectors are judged by the oracle as named and
#      exit 0 (accept_*) or 5 (reject_*).
#   3. N_FUZZ generated files: the exit status of spr1_check against the verdict
#      of formal/Bootstrap/SporosCheckDiff.lean, judged in one Lean process.
#      0 disagreements (accept <-> 0, reject <-> 5), at least 25% accepted, and
#      every file the generator built valid is accepted.
#   4. INJECT=1: single-fault injection from `tripled` (the input has been
#      triplicated) to the exit, on five invalid inputs (SDC_accept_invalid
#      must be 0) and on a valid one (counts reported). Faults: every bit of
#      the 16 general registers, CF/ZF/SF/OF, every bit of the instruction at
#      rip, every bit of the three descriptors and of the three input copies.
#   5. Negative controls, each of which the gate must catch:
#      (a) both guard branches and the status displacement of the accept stub
#          patched out: some reject vector exits 0;
#      (b) guard 1 and the displacement patched out, guard 2 intact: the reject
#          vectors still exit 5, and a single flag fault gives
#          SDC_accept_invalid > 0 (Linux x86-64 only).
#
# Addresses (`tripled`, `guard`, the descriptors at the start of passes p1, p2,
# p3) are read from the listing on every run, never hard-coded.
#
# Environment:
#   SPR1_JOBS            injection workers (default: all cores)
#   SPR1_INJECT_STRIDE   inject only every K-th step in step 4 (default 1: all)
#   FUZZ_SEED            generator seed (default 1)
#   ALLOW_EXIT6=1        count exit 6 as a rejection in step 3
#   SPR1_CHECK_BIN, SPR1_CHECK_HEX   test another binary and listing (step 1
#                        is then skipped)
set -euo pipefail

N_FUZZ="${1:-10000}"
INJECT="${2:-0}"
root="$(cd "$(dirname "$0")/../.." && pwd)"

: "${SPR1_JOBS:=0}"
: "${SPR1_INJECT_STRIDE:=1}"
: "${FUZZ_SEED:=1}"
: "${ALLOW_EXIT6:=0}"

lake="${LEAN_BIN:+$LEAN_BIN/}lake"
if ! command -v "$lake" >/dev/null 2>&1 && [[ -x "$HOME/.elan/bin/lake" ]]; then
  lake="$HOME/.elan/bin/lake"
fi
if [[ "$(uname -s)" == Darwin ]]; then cxx="${CXX:-clang++}"; else cxx="${CXX:-g++}"; fi

work="$(mktemp -d "${TMPDIR:-/tmp}/spr1_check_gate.XXXXXX")"
trap 'rm -rf "$work"' EXIT
work="$(cd "$work" && pwd)"

fail() { echo "spr1_check_gate: FAIL: $*" >&2; exit 1; }
say() { echo "spr1_check_gate: $*"; }

src_bin="${SPR1_CHECK_BIN:-$root/bootstrap/model/spr1_check}"
hex="${SPR1_CHECK_HEX:-$root/bootstrap/model/spr1_check.hex}"
[[ -f "$src_bin" ]] || fail "no binary $src_bin"
[[ -f "$hex" ]] || fail "no listing $hex"
bin="$work/spr1_check"  # an executable copy, whatever the mode of the checked-in file
cp "$src_bin" "$bin"
chmod 755 "$bin"

fuzz="$work/spr1_check_fuzz"
"$cxx" -std=c++23 -O2 -o "$fuzz" "$root/bootstrap/model/spr1_check_fuzz.cpp"
(cd "$root/formal" && "$lake" build Sporos.Run sporos-examples >/dev/null)
# from formal/, so that elan picks formal/lean-toolchain; every path passed is absolute
oracle() { (cd "$root/formal" && "$lake" env lean --run Bootstrap/SporosCheckDiff.lean "$@"); }

# ---- 1. reproduction from the listing
if [[ -n "${SPR1_CHECK_BIN:-}" ]]; then
  say "step 1 SKIPPED (SPR1_CHECK_BIN=$src_bin)"
else
  cp "$root/bootstrap/roots/a/seed-a" "$work/seed-a"
  chmod 755 "$work/seed-a"
  "$work/seed-a" < "$hex" > "$work/rebuilt" || fail "seed-a refused $hex"
  cmp "$work/rebuilt" "$src_bin" || fail "seed-a < spr1_check.hex differs from spr1_check"
  say "step 1 ok: seed-a < spr1_check.hex == spr1_check ($(wc -c < "$src_bin" | tr -d ' ') bytes)"
fi

"$fuzz" syms "$hex" "$bin" > "$work/syms.txt"
sym() { awk -v n="$1" '$2 == n { print $1; f = 1; exit } END { if (!f) exit 1 }' "$work/syms.txt" || fail "no symbol $1 in the listing"; }
tripled="$(sym tripled)"; guard="$(sym guard)"
desc1="$(sym desc1)"; desc2="$(sym desc2)"; desc3="$(sym desc3)"
inject_opts=(--desc copy1 "$desc1" 16 0 8 --desc copy2 "$desc2" 16 0 8 --desc copy3 "$desc3" 16 0 8
             --buf-extra 1 --jobs "$SPR1_JOBS")
say "symbols: tripled=$tripled guard=$guard descriptors=$desc1,$desc2,$desc3"

# exit status of `$1 < $2`, or sigN
status() { local rc=0; "$1" < "$2" > /dev/null 2>&1 || rc=$?; [[ $rc -gt 128 ]] && echo "sig$((rc - 128))" || echo "$rc"; }

# ---- 2. examples and fixed vectors
oracle gen "$work/ex_lean" >/dev/null
"$root/formal/.lake/build/bin/sporos-examples" "$work/ex_ref" >/dev/null
"$fuzz" examples "$work/ex_cpp"
for n in echo hello upper fib; do
  cmp "$work/ex_lean/$n.spb" "$work/ex_ref/$n.spb" || fail "example $n: SporosCheckDiff gen differs from sporos-examples"
  cmp "$work/ex_cpp/$n.spb" "$work/ex_ref/$n.spb" || fail "example $n: spr1_check_fuzz differs from sporos-examples"
  s="$(status "$bin" "$work/ex_ref/$n.spb")"
  [[ "$s" == 0 ]] || fail "example $n: exit $s, expected 0"
done
"$fuzz" vectors "$work/vec" >/dev/null
ls "$work"/vec/*.spb > "$work/vec.list"
oracle --list "$work/vec.list" > "$work/vec.oracle"
nacc=0; nrej=0
while read -r v f; do
  b="$(basename "$f")"
  case "$b" in
    accept_*) want=accept; code=0; nacc=$((nacc + 1)) ;;
    reject_*) want=reject; code=5; nrej=$((nrej + 1)) ;;
    *) fail "unexpected vector $b" ;;
  esac
  [[ "$v" == "$want" ]] || fail "vector $b: oracle says $v"
  s="$(status "$bin" "$f")"
  [[ "$s" == "$code" ]] || fail "vector $b: exit $s, expected $code"
done < "$work/vec.oracle"
[[ $((nacc + nrej)) == "$(wc -l < "$work/vec.list" | tr -d ' ')" ]] || fail "oracle output incomplete for the vectors"
say "step 2 ok: 4 examples exit 0 (identical from Lean, sporos-examples, C++); vectors accept=$nacc exit 0, reject=$nrej exit 5"

# ---- 3. differential fuzz
"$fuzz" gen "$N_FUZZ" "$FUZZ_SEED" "$work/fz" > "$work/gen.txt"
awk -v d="$work/fz" '{print d "/" $1}' "$work/fz/manifest.txt" > "$work/fz.list"
t0=$SECONDS
oracle --list "$work/fz.list" > "$work/fz.oracle"
t1=$SECONDS
"$fuzz" run "$bin" "$work/fz.list" 5000 > "$work/fz.status"
t2=$SECONDS
[[ "$(wc -l < "$work/fz.oracle" | tr -d ' ')" == "$N_FUZZ" ]] || fail "oracle judged $(wc -l < "$work/fz.oracle") of $N_FUZZ files"
[[ "$(wc -l < "$work/fz.status" | tr -d ' ')" == "$N_FUZZ" ]] || fail "ran $(wc -l < "$work/fz.status") of $N_FUZZ files"
awk -v allow6="$ALLOW_EXIT6" -v man="$work/fz/manifest.txt" -v st="$work/fz.status" '
  BEGIN {
    while ((getline l < man) > 0) { split(l, a, " "); kind[a[1]] = a[2] }
    while ((getline l < st) > 0) { split(l, a, " "); code[a[2]] = a[1] }
  }
  {
    v = $1; f = $2; n++; c = code[f]; b = f; sub(/.*\//, "", b)
    if (!(f in code)) { print "missing status for " f; bad++; next }
    if (v == "accept") acc++
    ok = (v == "accept" && c == "0") || (v == "reject" && (c == "5" || (allow6 == 1 && c == "6")))
    if (!ok) { bad++; if (bad <= 20) print "DISAGREE: oracle=" v " exit=" c " " b " (" kind[b] ")" }
    if (c == "6") e6++
    k = kind[b]
    if ((k == "valid" || k ~ /^example:/) && v != "accept") { genbad++; if (genbad <= 5) print "GENERATOR: built valid but rejected: " b }
  }
  END {
    printf "cases=%d accepts=%d (%.1f%%) disagreements=%d exit6=%d generator_errors=%d\n", n, acc, 100.0 * acc / n, bad, e6, genbad
    if (bad > 0 || genbad > 0 || 4 * acc < n) exit 1
  }' "$work/fz.oracle" > "$work/fz.cmp" || { cat "$work/fz.cmp"; fail "differential fuzz"; }
say "step 3 ok: $(tail -1 "$work/fz.cmp") [$(tr '\n' ' ' < "$work/gen.txt")] oracle $((t1 - t0))s, spr1_check $((t2 - t1))s"

# ---- 4. single-fault injection
can_inject=0
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] && can_inject=1
report() { grep -E '^(golden|total_injections)' "$1" | tr '\n' ' '; echo; grep '^class=' "$1" | sed 's/^/    /'; }
if [[ "$INJECT" == 1 ]]; then
  [[ $can_inject == 1 ]] || fail "INJECT=1 needs Linux on x86-64"
  total_sdc=0; total_inj=0
  for n in reject_main_ins reject_jump_wrong_height reject_opcode_8d reject_trailing_byte reject_ann_entry; do
    rc=0
    "$fuzz" inject "$bin" "$work/vec/$n.spb" "$tripled" --expect reject --stride "$SPR1_INJECT_STRIDE" \
      "${inject_opts[@]}" > "$work/inj_$n.txt" || rc=$?
    sdc="$(sed -n 's/^SDC_accept_invalid=//p' "$work/inj_$n.txt")"
    say "inject $n: $(report "$work/inj_$n.txt")"
    [[ $rc -le 1 && -n "$sdc" ]] || { cat "$work/inj_$n.txt"; fail "inject $n: harness error (rc $rc)"; }
    [[ "$sdc" == 0 ]] || { grep '^sdc:' "$work/inj_$n.txt"; fail "inject $n: SDC_accept_invalid=$sdc"; }
    total_sdc=$((total_sdc + sdc))
    total_inj=$((total_inj + $(sed -n 's/^total_injections=\([0-9]*\).*/\1/p' "$work/inj_$n.txt")))
  done
  rc=0
  "$fuzz" inject "$bin" "$work/vec/accept_tiny.spb" "$tripled" --expect accept --stride "$SPR1_INJECT_STRIDE" \
    "${inject_opts[@]}" > "$work/inj_valid.txt" || rc=$?
  [[ $rc == 0 ]] || { cat "$work/inj_valid.txt"; fail "inject accept_tiny: harness error (rc $rc)"; }
  say "inject accept_tiny (valid): $(report "$work/inj_valid.txt")"
  say "step 4 ok: SDC_accept_invalid=$total_sdc over $total_inj injections on 5 invalid inputs"
else
  say "step 4 SKIPPED (INJECT=0)"
fi

# ---- 5. negative controls
off() { printf '0x%x' $((guard + $1)); }
# (a) both guards and the accept stub's displacement
"$fuzz" patch "$bin" "$work/ctl_a" "$(off 0x12)" '75??>7500' "$(off 0x21)" '75??>7500' "$(off 0x2b)" '6affffff>ffffffff'
n0=0; nr=0
for f in "$work"/vec/reject_*.spb; do
  nr=$((nr + 1))
  if [[ "$(status "$work/ctl_a" "$f")" == 0 ]]; then n0=$((n0 + 1)); fi
done
[[ $n0 -gt 0 ]] || fail "negative control (a): no reject vector exits 0 on the patched binary"
say "step 5a ok: negative control PASSED (guards 1, 2 and the accept displacement patched: $n0 of $nr reject vectors exit 0, caught)"
# (b) guard 1 and the displacement; guard 2 alone must be beaten by one fault
"$fuzz" patch "$bin" "$work/ctl_b" "$(off 0x12)" '75??>7500' "$(off 0x2b)" '6affffff>ffffffff'
for f in "$work"/vec/reject_*.spb; do
  s="$(status "$work/ctl_b" "$f")"
  [[ "$s" == 5 ]] || fail "negative control (b): $(basename "$f") exits $s on the patched binary without a fault"
done
if [[ $can_inject == 1 ]]; then
  rc=0
  "$fuzz" inject "$work/ctl_b" "$work/vec/reject_main_ins.spb" "$tripled" --expect reject --classes flags \
    "${inject_opts[@]}" > "$work/ctl_b.txt" || rc=$?
  sdc="$(sed -n 's/^SDC_accept_invalid=//p' "$work/ctl_b.txt")"
  [[ $rc == 1 && -n "$sdc" && "$sdc" -gt 0 ]] || { cat "$work/ctl_b.txt"; fail "negative control (b): one flag fault did not beat guard 2"; }
  say "step 5b ok: negative control PASSED (guard 1 and the accept displacement patched: SDC_accept_invalid=$sdc, $(grep -m1 '^sdc:' "$work/ctl_b.txt"))"
else
  say "step 5b SKIPPED (injection needs Linux on x86-64)"
fi
say "PASS"
