#!/usr/bin/env bash
# Measure, for each test in tests/silent_verdicts.tsv, whether it currently
# prints its own success marker.
#
# These tests compute a verdict, print PASS or FAIL, and are judged by exit
# status alone -- and a `return` from a unit main exits 0, so a printed FAIL is
# counted green (#2150). Before annotating them with expect-stdout-contains,
# the honest order is to find out what they actually say. Annotating first and
# reading the CI failures afterwards would work, but it would land the answer
# as a regression rather than a measurement.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
MAD="${SOUNIO_SILENT_VERDICT_MADAROS:-}"
[[ -x "$MAD" ]] || { echo "SOUNIO_SILENT_VERDICT_MADAROS must name an executable Madaros ELF"; exit 2; }
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
W="$(mktemp -d /tmp/silent-verdicts.XXXXXX)"; trap 'rm -rf "$W"' EXIT
ulimit -s 524288 2>/dev/null || true
ok=0; bad=0; nobuild=0
while IFS=$'\t' read -r src marker; do
  [[ -n "$src" ]] || continue
  if ! timeout 300 "$MAD" build "$src" "$W/t.elf" >"$W/b.log" 2>&1; then
    printf '  NOBUILD  %-56s\n' "$(basename "$src")"; nobuild=$((nobuild+1)); continue
  fi
  chmod +x "$W/t.elf"
  timeout 180 "$W/t.elf" >"$W/r.log" 2>&1; rc=$?
  if grep -qF "$marker" "$W/r.log"; then
    printf '  SAYS-PASS  rc=%-3s %-50s\n' "$rc" "$(basename "$src")"; ok=$((ok+1))
  else
    printf '  SAYS-FAIL  rc=%-3s %-50s | %s\n' "$rc" "$(basename "$src")" "$(tail -2 "$W/r.log" | tr '\n' ' ' | cut -c1-60)"
    bad=$((bad+1))
  fi
done < tests/silent_verdicts.tsv
echo "silent_verdicts: says_pass=$ok says_fail=$bad nobuild=$nobuild"
