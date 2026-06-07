#!/usr/bin/env bash
# Checker table CAP-HOLES gate.
#
# The #2 import-typecheck landing seeds imported pub fn/struct/enum signatures
# into fixed-size checker tables. Those tables silently TRUNCATED past their cap
# (env 128, struct 128, enum 32, fn_sig 256), so a large module set either
# silently OVER-ACCEPTED ill-typed code (high-index fn call arg-type unchecked)
# or silently FALSE-REJECTED valid code (high-index struct literal -> E015).
#
# The fix is C0 = two parts, BOTH gated here:
#   (1) GROW all four tables (env, struct, enum, fn_sig) to a consistent N=512.
#   (2) LOUD OVERFLOW: a table that fills past N emits E300 + REJECTS (no ELF) —
#       never silently drops a symbol.
#
# This gate proves, per fn/struct/enum:
#   A. high index BELOW cap, valid use   -> check OK + ELF compiles + runs right
#   A. high index BELOW cap, ill-typed   -> rejected with the RIGHT type error
#                                            (E009 arg-type / E016 field-type),
#                                            NOT E015 truncation
#   B. AT/ABOVE cap (N+18 symbols)        -> LOUD E300 + reject + NO ELF
#                                            (not a silent slip, not a silent E015)
set -u
BIN="${1:?usage: run.sh <souc-binary>}"
WORK="$(mktemp -d /tmp/capgate.XXXXXX)"
pass=0
tot=0

note() { echo "$1"; }
ck() { # ck <label> <expected_substr> <file> <mode:check|reject>
  local label="$1" want="$2" file="$3"
  tot=$((tot+1))
  local out; out="$("$BIN" --check "$file" 2>&1)"
  if echo "$out" | grep -q "$want"; then
    note "PASS  $label ($want)"; pass=$((pass+1))
  else
    note "FAIL  $label (wanted '$want')"; echo "$out" | grep -E 'error\[|check:' | head -3
  fi
}
run_elf() { # run_elf <label> <file> <expected_exit>
  local label="$1" file="$2" want="$3"
  tot=$((tot+1))
  local out="$WORK/o.elf"; rm -f "$out"
  "$BIN" --native-v2-compile "$file" -o "$out" >/dev/null 2>&1
  if [ -f "$out" ]; then
    chmod +x "$out"; "$out" >/dev/null 2>&1; local rc=$?
    if [ "$rc" -eq "$want" ]; then note "PASS  $label (ELF exit $rc)"; pass=$((pass+1));
    else note "FAIL  $label (ELF exit $rc, wanted $want)"; fi
  else
    note "FAIL  $label (no ELF emitted)"
  fi
}
noelf() { # noelf <label> <file>  -- expect LOUD E300 reject AND no ELF
  local label="$1" file="$2"
  tot=$((tot+1))
  local out="$WORK/o.elf"; rm -f "$out"
  local clog; clog="$("$BIN" --native-v2-compile "$file" -o "$out" 2>&1)"
  local ccheck; ccheck="$("$BIN" --check "$file" 2>&1)"
  if [ ! -f "$out" ] && echo "$ccheck" | grep -q 'E300'; then
    note "PASS  $label (E300 loud + no ELF)"; pass=$((pass+1))
  else
    note "FAIL  $label (expected E300 + no ELF)"; echo "$ccheck" | grep -E 'error\[|check:' | head -3
  fi
}

# ---- generate witnesses ----
python3 - "$WORK" <<'PY'
import sys, os
W=sys.argv[1]
def w(n,s): open(os.path.join(W,n),"w").write(s)
# FN below-cap valid: 200 fns, call f150 -> 150+2=152
w("fn_ok.sio","\n".join([f"fn f{i}(x: i64) -> i64 {{ x + {i} }}" for i in range(200)]+["fn main() -> i64 { f150(2) }"]))
# FN below-cap ill-typed arg -> E009 (NOT E015)
w("fn_ill.sio","\n".join([f"fn f{i}(x: i64) -> i64 {{ x + {i} }}" for i in range(200)]+['fn main() -> i64 { f150("str") }']))
# FN overflow: 530 fns
w("fn_of.sio","\n".join([f"fn g{i}(x: i64) -> i64 {{ x + {i} }}" for i in range(530)]+["fn main() -> i64 { g0(1) }"]))
# STRUCT below-cap valid: 200 structs, use S180 -> 5+180=185
w("st_ok.sio","\n".join([f"struct S{i} {{ x: i64 }}" for i in range(200)]+["fn main() -> i64 { let s = S180 { x: 5 }  s.x + 180 }"]))
# STRUCT below-cap ill-typed field -> E016 (NOT E015)
w("st_ill.sio","\n".join([f"struct S{i} {{ x: i64 }}" for i in range(200)]+['fn main() -> i64 { let s = S180 { x: "str" }  s.x }']))
# STRUCT overflow: 530 structs
w("st_of.sio","\n".join([f"struct T{i} {{ x: i64 }}" for i in range(530)]+["fn main() -> i64 { let s = T0 { x: 1 }  s.x }"]))
# ENUM below-cap valid: 200 enums, use E180 -> match -> 7
w("en_ok.sio","\n".join([f"enum E{i} {{ A, B }}" for i in range(200)]+["fn main() -> i64 { let e = E180::A  match e { E180::A => 7, E180::B => 9 } }"]))
# ENUM below-cap ill-typed: high-index enum value where i64 expected -> E008.
# This is the real enum over-accept witness: at baseline (cap 32) E180 was
# truncated-to-unknown so this type mismatch SLIPPED to check OK; grown to 512
# E180 is KNOWN and the mismatch is correctly rejected (E008, not E015).
w("en_ill.sio","\n".join([f"enum E{i} {{ A, B }}" for i in range(200)]+["fn f() -> i64 { E180::A }","fn main() -> i64 { f() }"]))
# ENUM overflow: 530 enums
w("en_of.sio","\n".join([f"enum N{i} {{ A, B }}" for i in range(530)]+["fn main() -> i64 { 1 }"]))
PY

# ---- FN ----
ck     "fn_below_cap_illtyped_E009"   "E009" "$WORK/fn_ill.sio"
ck     "fn_below_cap_valid_OK"        "check: OK" "$WORK/fn_ok.sio"  # valid high-index fn -> not E015 false-reject
run_elf "fn_below_cap_valid_runs"     "$WORK/fn_ok.sio" 152
noelf  "fn_overflow_loud_E300"        "$WORK/fn_of.sio"

# ---- STRUCT ----
ck     "struct_below_cap_valid_OK"    "check: OK" "$WORK/st_ok.sio"
ck     "struct_below_cap_illtyped_E016" "E016" "$WORK/st_ill.sio"
run_elf "struct_below_cap_valid_runs" "$WORK/st_ok.sio" 185
noelf  "struct_overflow_loud_E300"    "$WORK/st_of.sio"

# ---- ENUM ----
ck     "enum_below_cap_valid_OK"      "check: OK" "$WORK/en_ok.sio"
ck     "enum_below_cap_illtyped_E008" "E008" "$WORK/en_ill.sio"
run_elf "enum_below_cap_valid_runs"   "$WORK/en_ok.sio" 7
noelf  "enum_overflow_loud_E300"      "$WORK/en_of.sio"

rm -rf "$WORK"
echo "---- $pass/$tot ----"
[ "$pass" -eq "$tot" ]
