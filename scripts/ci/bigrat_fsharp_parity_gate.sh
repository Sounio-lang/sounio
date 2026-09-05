#!/usr/bin/env bash
# ADR-009 verified_foreign_reference gate: F# for exact rational arithmetic.
#
# Sounio's stdlib/data/bigrat.sio implements BigRat (num/den in lowest
# terms, den > 0) over a from-scratch base-1e9-limb BigInt. The module's
# own CALIBRATION NOTE warns souc has "a whole-program codegen capacity
# wall for struct-heavy BigInt-by-value code" that can silently emit
# wrong values with a clean exit -- the existing scripts/bigrat_gate.sh
# already cross-checks against a Python oracle, but ADR-008 classifies
# any Python-authority path as external_corroboration_only (report-only,
# cannot fail CI regardless of how correct that Python happens to be).
#
# tools/fsharp/BigRatParity.fsx is an independently authored F#
# reference (System.Numerics.BigInteger, not a transliteration of
# Sounio's limb arithmetic) computing the same 9 test values. This gate
# runs both real implementations and diffs their output line for line;
# a mismatch is a hard CI failure, per ADR-009's verified_foreign_reference
# admission criteria (independent authorship, static typing / exactness,
# pinned toolchain, documented reason the Sounio twin isn't enough --
# BigRat has no independent second Sounio implementation to twin against).

set -euo pipefail
umask 077

fail() {
  printf 'bigrat-fsharp-parity: FAIL: %s\n' "$*" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOUC="${SOUNIO_TEST_SOUC_BIN:-$ROOT_DIR/bin/souc}"
SIO_CASE="$ROOT_DIR/tests/frontend/bigrat_fsharp_parity/cases.sio"
FSX_ORACLE="$ROOT_DIR/tools/fsharp/BigRatParity.fsx"
EXPECTED_DOTNET_MAJOR="10"

[[ -x "$SOUC" ]] || fail "souc not executable: $SOUC"
[[ -r "$SIO_CASE" ]] || fail "sounio case source not found: $SIO_CASE"
[[ -r "$FSX_ORACLE" ]] || fail "F# oracle source not found: $FSX_ORACLE"

command -v dotnet >/dev/null 2>&1 || \
  fail "dotnet not on PATH; install the .NET SDK (https://dotnet.microsoft.com/download), no sudo required via dotnet-install.sh --install-dir <dir>"

observed_major="$(dotnet --version 2>&1 | cut -d. -f1)"
[[ "$observed_major" == "$EXPECTED_DOTNET_MAJOR" ]] || \
  fail "dotnet major version drift: expected ${EXPECTED_DOTNET_MAJOR}.x, observed $(dotnet --version 2>&1)"

work="$(mktemp -d "${TMPDIR:-/tmp}/bigrat-fsharp-parity.XXXXXX")"
trap 'rm -rf "$work"' EXIT

"$SOUC" compile "$SIO_CASE" -o "$work/cases.elf" >"$work/sio_compile.log" 2>&1 || \
  fail "souc failed to compile $SIO_CASE: $(cat "$work/sio_compile.log")"
chmod +x "$work/cases.elf"
"$work/cases.elf" > "$work/sio_out.txt" 2>&1 || fail "sounio case binary exited non-zero"

dotnet fsi "$FSX_ORACLE" > "$work/fsx_out.txt" 2>"$work/fsx_stderr.log" || \
  fail "F# oracle execution failed: $(cat "$work/fsx_stderr.log")"

if ! diff -u "$work/sio_out.txt" "$work/fsx_out.txt" > "$work/diff.txt"; then
  cat "$work/diff.txt" >&2
  fail "Sounio bigrat output diverges from the independent F# reference"
fi

lines_checked="$(wc -l < "$work/sio_out.txt" | tr -d ' ')"

printf 'BIGRAT_FSHARP_PARITY_V1\n'
printf 'oracle_class=verified_foreign_reference\n'
printf 'producer_language=F#\n'
printf 'producer_role=EXACT_RATIONAL_ARITHMETIC_PARITY\n'
printf 'semantic_authority_language=Sounio\n'
printf 'dotnet_version=%s\n' "$(dotnet --version 2>&1)"
printf 'lines_checked=%s\n' "$lines_checked"
printf 'mismatches=0\n'
printf 'result=PASS\n'
