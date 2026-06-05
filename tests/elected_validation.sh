#!/usr/bin/env bash
# Rigorous validation of the elected modular compiler by EXIT CODE (never grep).
# Buckets: 0=OK, 1=reject, >=128=CRASH (139=SIGSEGV), parse-fail, other.
set -u
BIN="${1:?usage: elected_validation.sh <souc-binary>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
TMP=$(mktemp -d)

echo "############ 1) CAPGATE (codegen) ############"
bash "$ROOT/tests/native_v2_capgate/run.sh" "$BIN" 2>&1 | tail -1

echo "############ 2) POS/NEG CORRECTNESS (by exit code) ############"
mk(){ printf '%s' "$2" > "$TMP/$1.sio"; }
mk pos_valid   'fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(2, 3) }'
mk neg_typemis 'fn main() -> i64 { let x: i64 = true
 x }'
mk neg_undef   'fn main() -> i64 { undefined_fn(3) }'
mk neg_arity   'fn f(a: i64) -> i64 { a }
fn main() -> i64 { f(1, 2) }'
for n in pos_valid neg_typemis neg_undef neg_arity; do
  "$BIN" --check "$TMP/$n.sio" >/dev/null 2>&1; rc=$?
  case "$n" in pos_*) want=0;; neg_*) want=1;; esac
  verdict="OK"; [ "$rc" = "$want" ] || verdict="*** WRONG (rc=$rc want=$want) ***"
  printf "  %-12s exit=%-3s expect=%s  %s\n" "$n" "$rc" "$want" "$verdict"
done

echo "############ 3) EXIT-CODE CENSUS on real corpus ############"
ok=0; rej=0; crash=0; parse=0; other=0; tot=0
while read -r f; do
  [ -f "$ROOT/$f" ] || continue
  tot=$((tot+1))
  out=$(cd "$ROOT" && timeout 25 "$BIN" --check "$f" 2>&1); rc=$?
  if [ "$rc" -ge 128 ]; then crash=$((crash+1)); echo "  CRASH($rc) $f" >> "$TMP/crashes.txt"
  elif printf '%s' "$out" | grep -qi "parse"; then parse=$((parse+1))
  elif [ "$rc" = 0 ]; then ok=$((ok+1))
  elif [ "$rc" = 1 ]; then rej=$((rej+1))
  else other=$((other+1)); echo "  OTHER($rc) $f" >> "$TMP/other.txt"
  fi
done < <(cd "$ROOT" && git ls-files '*.sio' | grep -E '^(examples|tests/run-pass|tests/stdlib|stdlib)/')
echo "  total=$tot  OK=$ok  reject=$rej  CRASH=$crash  parse-fail=$parse  other=$other"
echo "  --- crashes (if any) ---"; [ -f "$TMP/crashes.txt" ] && head -20 "$TMP/crashes.txt" || echo "  none"
echo "VALIDATION DONE. tmp=$TMP"
