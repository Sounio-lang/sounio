#!/usr/bin/env bash
# The lean_single seed refuses a source with more tokens than its buffer holds.
#
# Before this gate, lex_all() stopped at the buffer's capacity and the parser
# ran on the truncated stream. The result was a misleading
# "parse error: expected expression ... found ``" at a line with no error.
# That is how #3506 failed: the compiler tree on main used 2,094,834 of the
# 2,097,152 tokens, and the SCO private-controls copy of #3506 needed 2,097,710.
# The capacity is now 4,194,304 tokens, and overflow is an explicit refusal.
#
# The input is one array literal with 4,400,001 elements, about 8.8 million
# tokens in 8.8 MB, so the 16 MiB source buffer is not what refuses it. The
# gate requires the token-buffer refusal and no parse error.
# Measured on the seed before the fix: rc=1 with "parse error: expected
# expression at <main>:2". After: rc=1 with the refusal below.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${SOUNIO_LEAN_SINGLE_BIN:-$ROOT/bin/souc-lean-single-x86_64}"
WORK="${1:-$(mktemp -d "${TMPDIR:-/tmp}/lean-single-token-capacity.XXXXXX")}"
mkdir -p "$WORK"
[[ -x "$SEED" ]] || { echo "FAIL: missing lean_single seed: $SEED" >&2; exit 1; }
src="$WORK/too_many_tokens.sio"
{
  printf 'fn main() -> i64 {\n    let x = ['
  head -c 4400000 /dev/zero | tr '\0' '0' | sed 's/0/0,/g'
  printf '0]\n    0\n}\n'
} > "$src"
rc=0
( ulimit -s unlimited 2>/dev/null || true
  cd "$WORK" && timeout 600 "$SEED" "$src" "$WORK/too_many_tokens.elf" ) > "$WORK/seed.log" 2>&1 || rc=$?
if [[ $rc == 0 ]]; then
  echo "FAIL: the seed accepted a source larger than its token buffer" >&2; exit 1
fi
if grep -q 'parse error' "$WORK/seed.log"; then
  echo "FAIL: token overflow surfaced as a parse error, not a refusal:" >&2
  grep -m3 'parse error\|tokens:' "$WORK/seed.log" >&2; exit 1
fi
# The seed's print_int ends its own line, so match the message with newlines folded.
tr '\n' ' ' < "$WORK/seed.log" | grep -Eq 'error: source too large: the token buffer of 4194304 +tokens filled before the end of the source' \
  || { echo "FAIL: missing token-buffer refusal (rc=$rc):" >&2; tail -5 "$WORK/seed.log" >&2; exit 1; }
echo "LEAN_SINGLE_TOKEN_CAPACITY_GATE_PASS rc=$rc"
