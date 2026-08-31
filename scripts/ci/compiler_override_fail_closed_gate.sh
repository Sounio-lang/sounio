#!/usr/bin/env bash
# compiler_override_fail_closed_gate.sh — naming a compiler must select it or stop.
#
# WHY. `bin/souc` and `bin/madaros` resolve the raw ELF by walking a candidate
# list and taking the first that is executable. An override that is SET but not
# executable failed that test and was SKIPPED, so resolution continued to the
# COMMITTED bin/madaros-linux-x86_64 — which lags self-hosted/ source. The run
# then reported on a compiler the caller did not name, exit 0, no message.
#
# Measured 2026-08-28, same file, same command, one `chmod` apart:
#   chmod 600 elf; MADAROS_RAW_BIN=$elf souc check f.sio  ->  "check: OK"   rc=0
#   chmod 700 elf; MADAROS_RAW_BIN=$elf souc check f.sio  ->  "error[E245]" rc!=0
#
# Opposite verdicts about the language, decided by a permission bit. This gate
# drives BOTH directions: the refusal must fire, and the working paths must
# still work — a guard that rejects everything would pass a one-sided test.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9

W="$(mktemp -d /tmp/override_fail_closed.XXXXXX)"
trap 'rm -rf "$W"' EXIT
printf 'fn main() -> i64 with IO {\n    println("x")\n    0\n}\n' > "$W/t.sio"

REAL="$ROOT_DIR/bin/madaros-linux-x86_64"
[[ -f "$REAL" ]] || { echo "GATE SKIP: no committed madaros ELF to derive fixtures from"; exit 0; }
cp "$REAL" "$W/noexec.elf"; chmod 600 "$W/noexec.elf"
cp "$REAL" "$W/ok.elf";     chmod 700 "$W/ok.elf"
printf '#!/bin/sh\necho nope\n' > "$W/script.elf"; chmod 755 "$W/script.elf"
cp "$REAL" "$W/local.elf"; printf '\0' >> "$W/local.elf"; chmod 700 "$W/local.elf"   # byte-different, still runs: a "local build"

fails=0
check() {  # check <name> <expect-rc0|expect-reject> <needle> <cmd...>
  local name="$1" expect="$2" needle="$3"; shift 3
  local out rc
  out="$(env -u SOUC_BIN SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$@" 2>&1)"; rc=$?
  if [[ "$expect" == "expect-reject" ]]; then
    if [[ $rc -eq 0 ]]; then
      echo "  FAIL $name — exited 0. It fell back to another compiler in silence." >&2
      echo "$out" | tail -3 | sed 's/^/       /' >&2; fails=$((fails+1)); return
    fi
    if [[ -n "$needle" ]] && ! grep -qF -- "$needle" <<<"$out"; then
      echo "  FAIL $name — refused (rc=$rc) but never said '$needle'." >&2
      echo "$out" | tail -3 | sed 's/^/       /' >&2; fails=$((fails+1)); return
    fi
  else
    if [[ $rc -ne 0 ]]; then
      echo "  FAIL $name — a usable configuration was rejected (rc=$rc)." >&2
      echo "$out" | tail -3 | sed 's/^/       /' >&2; fails=$((fails+1)); return
    fi
  fi
  echo "  ok   $name"
}

echo "[override-fail-closed] refusals must fire:"
check "souc: non-executable MADAROS_RAW_BIN" expect-reject "not executable" \
  env MADAROS_RAW_BIN="$W/noexec.elf" ./bin/souc check "$W/t.sio"
check "souc: missing MADAROS_RAW_BIN" expect-reject "no such file" \
  env MADAROS_RAW_BIN="$W/absent.elf" ./bin/souc check "$W/t.sio"
check "souc: MADAROS_RAW_BIN is a script" expect-reject "not a raw ELF" \
  env MADAROS_RAW_BIN="$W/script.elf" ./bin/souc check "$W/t.sio"
check "souc: non-executable SOUNIO_SOUC_BIN" expect-reject "not executable" \
  env SOUNIO_SOUC_BIN="$W/noexec.elf" ./bin/souc check "$W/t.sio"
check "madaros: non-executable MADAROS_RAW_BIN" expect-reject "not executable" \
  env MADAROS_RAW_BIN="$W/noexec.elf" ./bin/madaros check "$W/t.sio"
check "souc: strict mode refuses a non-committed ELF"    expect-reject "not the committed" \
  env SOUNIO_REQUIRE_COMMITTED_MADAROS=1 MADAROS_RAW_BIN="$W/local.elf" ./bin/souc --version
check "madaros: strict mode refuses a non-committed ELF" expect-reject "not the committed" \
  env SOUNIO_REQUIRE_COMMITTED_MADAROS=1 MADAROS_RAW_BIN="$W/local.elf" ./bin/madaros --version

# The same defect lives in two sourced libraries, and they are the wider door:
# scripts/lib/resolve_souc.sh is sourced by 126 scripts. They are checked here
# rather than in a gate of their own, because the question is identical and a
# second gate asking it separately is how one of them gets fixed and the other
# does not.
lib_check() {  # lib_check <name> <lib> <var> <value> <expect-refusal|expect-ok>
  local name="$1" lib="$2" var="$3" val="$4" expect="$5" out
  out="$(env -u SOUC_BIN -u MADAROS_BIN -u SOUNIO_MADAROS_BIN "$var=$val" \
         bash -c "source $lib >/dev/null 2>>'$W/lib.err'; echo \"BIN=[\${SOUC_BIN:-}\${MADAROS_BIN:-}]\"" 2>&1)"
  local err; err="$(cat "$W/lib.err" 2>/dev/null)"; : > "$W/lib.err"
  if [[ "$expect" == "expect-refusal" ]]; then
    if ! grep -qF "is set but cannot be used" <<<"$err"; then
      echo "  FAIL $name — no refusal on stderr." >&2; fails=$((fails+1)); return
    fi
    if grep -qF "BIN=[$val]" <<<"$out" || [[ "$out" == *"BIN=[]"* ]]; then :; else
      echo "  FAIL $name — a different binary was substituted: $out" >&2; fails=$((fails+1)); return
    fi
  else
    if [[ "$out" != *"BIN=[$val]"* ]]; then
      echo "  FAIL $name — a usable override was not honoured: $out" >&2; fails=$((fails+1)); return
    fi
  fi
  echo "  ok   $name"
}

: > "$W/lib.err"
echo "[override-fail-closed] sourced resolver libraries:"
lib_check "resolve_souc: unusable SOUC_BIN refused"        scripts/lib/resolve_souc.sh    SOUC_BIN    "$W/noexec.elf" expect-refusal
lib_check "resolve_souc: valid SOUC_BIN honoured"          scripts/lib/resolve_souc.sh    SOUC_BIN    "$W/ok.elf"     expect-ok
lib_check "resolve_madaros: unusable MADAROS_BIN refused"  scripts/lib/resolve_madaros.sh MADAROS_BIN "$W/noexec.elf" expect-refusal
lib_check "resolve_madaros: valid MADAROS_BIN honoured"    scripts/lib/resolve_madaros.sh MADAROS_BIN "$W/ok.elf"     expect-ok

echo "[override-fail-closed] working configurations must still work:"
check "no override resolves normally"   expect-rc0 "" ./bin/souc check "$W/t.sio"
check "valid override is honoured"      expect-rc0 "" env MADAROS_RAW_BIN="$W/ok.elf" ./bin/souc check "$W/t.sio"
check "empty override is not an override" expect-rc0 "" env MADAROS_RAW_BIN= ./bin/souc check "$W/t.sio"
check "--version unaffected"            expect-rc0 "" ./bin/souc --version
check "strict mode honours the committed ELF by content" expect-rc0 "" \
  env SOUNIO_REQUIRE_COMMITTED_MADAROS=1 MADAROS_RAW_BIN="$W/ok.elf" ./bin/souc --version
# provenance must TELL THE TRUTH about a local build (measured 2026-08-31, #2318)
_pv="$(env -u SOUC_BIN MADAROS_RAW_BIN="$W/local.elf" ./bin/souc --version 2>&1)"
if grep -q "LOCAL BUILD" <<<"$_pv" && ! grep -q "is the COMMITTED binary" <<<"$_pv"; then
  echo "  ok   --version names a local build honestly"
else
  echo "  FAIL --version called a non-committed ELF COMMITTED (or stayed silent)" >&2; fails=$((fails+1))
fi
_pv="$(env -u SOUC_BIN MADAROS_RAW_BIN="$W/ok.elf" ./bin/souc --version 2>&1)"
if grep -q "is the COMMITTED binary" <<<"$_pv"; then
  echo "  ok   --version recognises the committed ELF by content"
else
  echo "  FAIL --version did not recognise a byte-identical copy as committed" >&2; fails=$((fails+1))
fi

if [[ $fails -gt 0 ]]; then
  echo >&2
  echo "COMPILER_OVERRIDE_FAIL_CLOSED_GATE: $fails case(s) wrong." >&2
  echo "  Naming a compiler that cannot be used must stop the run. Falling through" >&2
  echo "  to the committed ELF answers a question nobody asked, and exits 0." >&2
  exit 1
fi
echo "COMPILER_OVERRIDE_FAIL_CLOSED_GATE_OK: 18 cases, 9 of them refusals, each behaved as stated"
