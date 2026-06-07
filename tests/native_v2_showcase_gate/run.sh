#!/usr/bin/env bash
# native_v2_showcase_gate — dogfoods the release showcase programs end-to-end through the
# modular compiler: each must --check clean, --native-v2-compile to an ELF, and RUN to its
# documented exit code. These are the import-free "verified native subset" demos
# (examples/showcase/README.native_v2.md). If a future compiler change breaks one, this
# fails loudly rather than letting the flagship demo silently rot.
set -u
BIN="${1:?usage: run.sh <souc-binary>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ulimit -s 1048576 2>/dev/null || true

# program:expected_exit
CASES="prime_sieve_100:25 tiny_register_vm:42"

pass=0; tot=0
for c in $CASES; do
  tot=$((tot+1))
  n=${c%:*}; exp=${c#*:}
  src="$ROOT/examples/showcase/$n.sio"
  out="/tmp/showcase_$n.elf"
  rm -f "$out"
  if ! "$BIN" --check "$src" >/dev/null 2>&1; then
    echo "FAIL  $n (--check rejected)"; continue
  fi
  "$BIN" --native-v2-compile "$src" -o "$out" >/dev/null 2>&1
  if [ ! -f "$out" ]; then echo "FAIL  $n (no ELF emitted)"; continue; fi
  chmod +x "$out"; "$out" >/dev/null 2>&1; rc=$?
  if [ "$rc" = "$exp" ]; then echo "PASS  $n (exit $rc)"; pass=$((pass+1));
  else echo "WRONG $n (got exit $rc want $exp)"; fi
done
echo "---- $pass/$tot ----"
[ "$pass" = "$tot" ] || exit 1
