#!/usr/bin/env bash
# #2533 / #2534: Madaros dynlink tier caps must never drop a library or a
# string silently.
#
#  1. tests/run-pass/dynlink_five_needed.sio -- five distinct libraries under
#     the symbol cap. Before #2533 native_v2_dynlink_unique_lib_count stopped
#     at 4, the cap check in the writer was dead, the build exited 0 and
#     DT_NEEDED libzstd.so.1 was missing (load-time "undefined symbol").
#     Required: >= 5 DT_NEEDED entries including libzstd.so.1, and the program
#     runs and prints PASS.
#  2. tests/run-pass/dynlink_all_symbols_dynstr.sio -- all 13 symbols / 6
#     libraries the emitter knows. Before the fix it was refused at the
#     8-symbol cap (rc 22, the same rc as the label-overflow refusal) and
#     .dynstr was a fixed 256-byte block with unchecked writes (#2534).
#     Required: it builds and runs, every NEEDED/dynsym name is present, and
#     DT_STRSZ equals exactly 1 + sum(len+1) over the NEEDED names and the
#     dynamic symbol names (no fixed reservation, no overlap).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TAG="madaros_dynlink_caps_gate"
fail() { echo "$TAG: FAIL $*" >&2; exit 1; }

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-dyncaps.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-${MADAROS_BIN:-}}}"
[[ -x "$SOUC" ]] || { echo "$TAG: souc not executable: $SOUC" >&2; exit 2; }
[[ -n "$RAW" && -x "$RAW" ]] || { echo "$TAG: set MADAROS_RAW_BIN to a Madaros ELF" >&2; exit 2; }
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

gcc -shared -fPIC -O0 -o "$WORKDIR/libkl14b_probe.so" tests/fixtures/kl14b/kl14b_probe.c
gcc -shared -fPIC -O0 -o "$WORKDIR/libkl14c_probe.so" tests/fixtures/kl14c/kl14c_probe.c
gcc -shared -fPIC -O0 -o "$WORKDIR/libkl14d_a.so" tests/fixtures/kl14d/lib_a.c
gcc -shared -fPIC -O0 -o "$WORKDIR/libkl14d_b.so" tests/fixtures/kl14d/lib_b.c
if ! ldconfig -p 2>/dev/null | grep -q 'libzstd.so.1 '; then
  Z="$(find /usr/lib /lib /usr/lib64 -name 'libzstd.so.1*' 2>/dev/null | head -1 || true)"
  [[ -n "$Z" ]] || { echo "$TAG: libzstd.so.1 not found (apt-get install libzstd1)" >&2; exit 2; }
  cp -Lf "$Z" "$WORKDIR/libzstd.so.1"
fi
export LD_LIBRARY_PATH="$WORKDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

build_and_run() {  # <fixture> -> ELF at $WORKDIR/<base>.elf, checks PASS line
  local src="tests/run-pass/$1.sio" out="$WORKDIR/$1.elf"
  if ! MADAROS_RAW_BIN="$RAW" "$SOUC" build "$src" -o "$out" >"$WORKDIR/$1.c.out" 2>&1; then
    cat "$WORKDIR/$1.c.out" >&2
    fail "$1: compile refused"
  fi
  chmod +x "$out"
  local got rc=0
  got="$("$out" 2>"$WORKDIR/$1.run.err")" || rc=$?
  if [[ $rc -ne 0 ]] || ! grep -qxF "$1: PASS" <<<"$got"; then
    cat "$WORKDIR/$1.run.err" >&2 || true
    readelf -d "$out" >&2 || true
    fail "$1: run rc=$rc stdout=[$got]"
  fi
}

needed_of() { readelf -d "$1" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p'; }

# ---- 1. five DT_NEEDED libraries (#2533)
build_and_run dynlink_five_needed
mapfile -t needed < <(needed_of "$WORKDIR/dynlink_five_needed.elf")
[[ ${#needed[@]} -ge 5 ]] || fail "dynlink_five_needed: ${#needed[@]} DT_NEEDED entries (${needed[*]}), expected 5"
printf '%s\n' "${needed[@]}" | grep -qxF libzstd.so.1 || fail "dynlink_five_needed: DT_NEEDED libzstd.so.1 dropped (${needed[*]})"
echo "$TAG: PASS five_needed (${needed[*]})"

# ---- 2. every known symbol, exact .dynstr (#2533 symbol cap, #2534)
build_and_run dynlink_all_symbols_dynstr
elf="$WORKDIR/dynlink_all_symbols_dynstr.elf"
mapfile -t needed < <(needed_of "$elf")
[[ ${#needed[@]} -eq 6 ]] || fail "all_symbols: ${#needed[@]} DT_NEEDED entries (${needed[*]}), expected 6"
mapfile -t syms < <(readelf -W --use-dynamic -s "$elf" | awk '$1 ~ /^[0-9]+:$/ && $8 != "" {print $8}')
[[ ${#syms[@]} -eq 13 ]] || fail "all_symbols: ${#syms[@]} dynamic symbols (${syms[*]}), expected 13"
want=1
for n in "${needed[@]}" "${syms[@]}"; do want=$(( want + ${#n} + 1 )); done
strsz="$(readelf -d "$elf" | awk '/\(STRSZ\)/ {print $3}')"
[[ "$strsz" == "$want" ]] || fail "all_symbols: DT_STRSZ=$strsz, expected exactly $want (sized from content)"
echo "$TAG: PASS all_symbols (6 NEEDED, 13 symbols, DT_STRSZ=$strsz)"

echo "MADAROS_DYNLINK_CAPS_GATE_OK"
