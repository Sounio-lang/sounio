#!/usr/bin/env bash
# stage0 fails closed (ADR-004): a program outside the subset that
# bootstrap/stage0.c compiles is refused (exit 2, no output file), never
# compiled into a silently wrong ELF.
#
# Before this gate, stage0 dropped what it could not handle and exited 0:
# unknown identifiers and enum variants compiled as 0, calls to unknown
# functions as `call +0`, every variant/literal match arm as taken, every
# struct field store as a store to field 0, global initializers as zero,
# and code past 1 MiB or functions past 1024 were cut off. Built that way,
# lean_single.sio came out as a 1 MiB ELF that segfaulted.
#
# Each refusal probe below compiled with exit 0 before the fix. The field
# store probe printed `7` and `2` before the fix (the store hit field a).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
CC_BIN="${CC:-cc}"

echo "== stage0_fail_closed_gate =="
"$CC_BIN" -O2 -Wno-unused-function -Wno-unused-variable -o "$OUT/stage0" "$ROOT/bootstrap/stage0.c"

fail=0
expect_refused() {
  local name="$1" what="$2" rc=0
  "$OUT/stage0" "$OUT/$name.sio" "$OUT/$name.elf" > /dev/null 2> "$OUT/$name.log" || rc=$?
  if [[ "$rc" -ne 2 ]] || ! grep -q "stage0: refused: $what" "$OUT/$name.log" || [[ -e "$OUT/$name.elf" ]]; then
    echo "FAIL: $name: expected exit 2, 'refused: $what' and no output file; got rc=$rc"
    tail -3 "$OUT/$name.log" || true
    fail=1
  else
    echo "ok   $name refused ($what)"
  fi
}

cat > "$OUT/unknown_ident.sio" <<'P'
fn main() -> i64 {
    let x = NOT_DEFINED
    print_int(x)
    0
}
P
expect_refused unknown_ident "identifier NOT_DEFINED"

cat > "$OUT/unknown_call.sio" <<'P'
fn main() -> i64 {
    let x = not_defined(1)
    print_int(x)
    0
}
P
expect_refused unknown_call "call to unknown function not_defined"

cat > "$OUT/match_literal.sio" <<'P'
fn main() -> i64 {
    let x = 2
    let y = match x {
        1 => 10,
        _ => 20,
    }
    print_int(y)
    0
}
P
expect_refused match_literal "match on a variant or literal pattern"

cat > "$OUT/global_init.sio" <<'P'
var CAP: i64 = 16
fn main() -> i64 {
    print_int(CAP)
    0
}
P
expect_refused global_init "global CAP"

{ for i in $(seq 1 1100); do printf 'fn f%d() -> i64 { %d }\n' "$i" "$i"; done
  printf 'fn main() -> i64 {\n    print_int(f1())\n    0\n}\n'; } > "$OUT/many_fns.sio"
expect_refused many_fns "capacity exceeded: functions"

cat > "$OUT/dup_method.sio" <<'P'
struct A { x: i64 }
struct B { y: i64 }
impl A { fn get(self) -> i64 { self.x } }
impl B { fn get(self) -> i64 { self.y } }
fn main() -> i64 {
    print_int(1)
    0
}
P
expect_refused dup_method "method B::get"

# The zero initializers stay accepted.
cat > "$OUT/global_zero.sio" <<'P'
var N: i64 = 0
var BUF: [i8; 64] = [0; 64]
fn main() -> i64 {
    print_int(N)
    print("\n")
    0
}
P
if "$OUT/stage0" "$OUT/global_zero.sio" "$OUT/global_zero.elf" > /dev/null 2> "$OUT/global_zero.log"; then
  echo "ok   global_zero accepted"
else
  echo "FAIL: zero global initializers refused"; tail -3 "$OUT/global_zero.log"; fail=1
fi

# A struct field store lands on the named field.
cat > "$OUT/field_store.sio" <<'P'
struct P { a: i64, b: i64 }
fn main() -> i64 {
    var p = P { a: 1, b: 2 }
    p.b = 7
    print_int(p.a)
    print("\n")
    print_int(p.b)
    print("\n")
    0
}
P
if "$OUT/stage0" "$OUT/field_store.sio" "$OUT/field_store.elf" > /dev/null 2> "$OUT/field_store.log"; then
  got="$("$OUT/field_store.elf" | tr '\n' ' ')"
  if [[ "$got" == "1 7 " ]]; then echo "ok   field_store prints 1 7"; else echo "FAIL: field_store printed '$got', expected '1 7 '"; fail=1; fi
else
  echo "FAIL: field_store did not compile"; tail -3 "$OUT/field_store.log"; fail=1
fi

# The bootstrap chain stage0 exists for still works: boot1 compiles hello.
"$OUT/stage0" "$ROOT/bootstrap/boot1.sio" "$OUT/boot1.elf" > /dev/null 2> "$OUT/boot1.log" \
  || { echo "FAIL: stage0 refused boot1.sio"; tail -3 "$OUT/boot1.log"; fail=1; }
if [[ -x "$OUT/boot1.elf" ]]; then
  printf 'fn main() -> i64 {\n    print("BOOTSTRAP OK\\n")\n    0\n}\n' > "$OUT/hello.sio"
  "$OUT/boot1.elf" "$OUT/hello.sio" "$OUT/hello.elf" > /dev/null 2>&1 && chmod +x "$OUT/hello.elf" \
    && [[ "$("$OUT/hello.elf")" == "BOOTSTRAP OK" ]] && echo "ok   boot1 compiles hello" \
    || { echo "FAIL: boot1 built by stage0 does not compile hello"; fail=1; }
fi

[[ "$fail" -eq 0 ]] || exit 1
echo "STAGE0_FAIL_CLOSED_GATE_OK"
