#!/usr/bin/env bash
# Inspect a function range identified from the build's function map.
# This runs objdump only; it never executes the compiler under investigation.
set -euo pipefail
binary="${1:?raw ELF}" start="${2:?function start VA}" stop="${3:?next function VA}" report="${4:?report path}"
[[ "$start" =~ ^0x[[:xdigit:]]+$ && "$stop" =~ ^0x[[:xdigit:]]+$ ]] || exit 64
sha256sum "$binary" > "$report.compiler.sha256"
objdump -D -b binary -m i386:x86-64 --adjust-vma=0x400000 \
  --start-address="$start" --stop-address="$stop" "$binary" > "$report"
if grep -Eq "[[:space:]]call[[:space:]]+$start([[:space:]]|$)" "$report"; then
 echo "FAIL: function at $start calls itself" >&2
 exit 1
fi
echo "PASS: function at $start has no direct self-call"
