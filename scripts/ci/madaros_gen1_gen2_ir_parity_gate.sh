#!/usr/bin/env bash
# gen1 and gen2 must lower the same programs to the same IR.
#
# gen2 = gen1(main.sio) and gen3 = gen2(main.sio), so gen2 == gen3 needs gen1
# and gen2 to BEHAVE the same on the compiler's own source. gen1 is built by
# lean_single and gen2 by Madaros: every construct the two compile differently
# shows up here, on small programs, minutes after gen2 exists, instead of as an
# unexplained byte difference after the two-hour gen3 rung (#2966). Each
# program below pinned one such class when this gate was written:
#   gum_euler_ode / gum_fo_across_call  `let s = *lo.fo_bind` shared its arrays
#                                       on Madaros, so the inline-variance
#                                       restore restored nothing in gen2;
#   a14_transitive_min                  lean_single gave a `use` item an empty
#                                       IR slot in gen1 (preseed dispatch);
#   algebra_decl_basic / effect_handler_basic  the same for `algebra` / `effect`.
# The comparison is SOUNIO_NV2_IR_TRACE (every IR instruction of every function).
#
# Usage: madaros_gen1_gen2_ir_parity_gate.sh GEN1 GEN2
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GEN1="${1:?usage: $0 GEN1 GEN2}"
GEN2="${2:?usage: $0 GEN1 GEN2}"
WORK="$(mktemp -d /tmp/sounio-ir-parity.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-ir-parity] FAIL: $*" >&2; exit 1; }
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576 2>/dev/null || true
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"

PROGRAMS=(gum_euler_ode gum_fo_across_call a14_transitive_min algebra_decl_basic effect_handler_basic)
compared=0
for p in "${PROGRAMS[@]}"; do
  src="$ROOT_DIR/tests/run-pass/$p.sio"
  [ -f "$src" ] || fail "$p: missing $src"
  for g in 1 2; do
    bin="$GEN1"; [ "$g" = 2 ] && bin="$GEN2"
    SOUNIO_NV2_IR_TRACE=1 "$bin" build "$src" -o "$WORK/$p.$g.elf" >"$WORK/$p.$g.log" 2>&1 \
      || fail "$p: gen$g failed to build it: $(tail -2 "$WORK/$p.$g.log")"
    grep '^NV2_IR' "$WORK/$p.$g.log" >"$WORK/$p.$g.tr" || true
    n="$(wc -l <"$WORK/$p.$g.tr")"
    [[ "$n" -ge 10 ]] || fail "$p: gen$g printed $n IR trace lines (expected >= 10); nothing to compare"
  done
  if ! cmp -s "$WORK/$p.1.tr" "$WORK/$p.2.tr"; then
    first="$( { diff "$WORK/$p.1.tr" "$WORK/$p.2.tr" || true; } | head -4 | tr "\n" " ")"
    fn="$( { diff "$WORK/$p.1.tr" "$WORK/$p.2.tr" || true; } | grep -m1 -o "name=[A-Za-z_0-9]*" || true)"
    fail "$p: gen1 and gen2 lower it differently (${fn:-module header}): $first"
  fi
  compared=$((compared + 1))
done
[[ "$compared" -ge 5 ]] || fail "compared $compared programs, expected 5"
echo "[madaros-ir-parity] PASS: gen1 and gen2 lower $compared programs to identical IR"
